bl_info = {
    "name": "Animotion3D - AI Director",
    "author": "Saul Heriberto Rodriguez Barajas",
    "version": (0, 6),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Animotion3D",
    "description": "Genera animaciones con IA local (LM Studio). Bones, objetos, cámaras, luces. Feedback loop, rig mapping y exportación SFT.",
    "category": "Animation",
}

import bpy
import json
import math
import threading
import time
import os
import urllib.request
import urllib.error
from typing import Any, Optional
from mathutils import Euler, Quaternion
from bpy.props import (
    StringProperty, CollectionProperty,
    IntProperty, FloatProperty, BoolProperty,
)
from bpy.types import PropertyGroup, Operator, Panel, UIList, UILayout

# ══════════════════════════════════════════════════════
#  CONFIG / DEFAULTS
# ══════════════════════════════════════════════════════
DEFAULT_URL     = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL   = "deepseek/deepseek-r1-0528-qwen3-8b"
DEFAULT_TIMEOUT = 300
DEFAULT_RETRIES = -1   # -1 = reintentos infinitos hasta cancelar
DEFAULT_BACKOFF = 1.0   # segundos base para backoff exponencial
MAX_BACKOFF_WAIT = 30.0  # límite para que el backoff no crezca sin control
MAX_CONV_TURNS  = 10     # máx turnos en historial (evita prompts gigantes)
SOCKET_TIMEOUT  = 10.0   # timeout de conexión TCP para detección rápida de cancel
MAX_LOG_ENTRIES = 60
ANIMATABLE_OBJECT_TYPES = {"ARMATURE", "CAMERA", "LIGHT", "MESH", "CURVE", "EMPTY"}
SUPPORTED_INTERPOLATION_MODES = {"BEZIER", "LINEAR", "CONSTANT"}
EULER_ROTATION_MODES = {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"}

DATASET_DIR  = os.path.join(os.path.expanduser("~"), "animotion3d_datasets")
SFT_FILENAME = os.path.join(DATASET_DIR, "sft_pairs.jsonl")
os.makedirs(DATASET_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════
SYSTEM_PROMPT = """Blender animation assistant. Respond ONLY with raw JSON, no markdown, no explanations.

KEYFRAME TYPES:
{"type":"bone","frame":1,"bone":"<name>","object":"<armature>","loc":[x,y,z],"rot":[xd,yd,zd],"scale":[x,y,z]}
{"type":"object","frame":1,"object":"<name>","loc":[x,y,z],"rot":[xd,yd,zd],"scale":[x,y,z]}
{"type":"property","frame":1,"object":"<name>","property":"data.lens","value":50.0}

BEHAVIOR TYPES:
{"type":"set_frame_range","start":1,"end":120}
{"type":"set_scene_fps","fps":24}
{"type":"set_interpolation","mode":"BEZIER|LINEAR|CONSTANT","object":"<name>|*"}
{"type":"set_active_camera","frame":1,"object":"<camera>"}

OUTPUT FORMAT:
{"animations":[...],"behaviors":[...],"notes":"<brief>"}

RULES:
- frame >= 1, rot in DEGREES
- Use SPARSE keyframes only: place keyframes only at key poses (start, peak, end of each movement). Let Blender interpolate between them. Do NOT place a keyframe on every frame.
- Aim for 3-8 keyframes per movement arc, not one per frame.
- Use EXACT bone/object names from scene list
- Never animate objects outside the provided context list
"""

# ══════════════════════════════════════════════════════
#  PROPERTY GROUPS
# ══════════════════════════════════════════════════════
class AM3D_ConvMessage(PropertyGroup):
    role:    StringProperty()
    content: StringProperty()

class AM3D_LogItem(PropertyGroup):
    level:   StringProperty()
    message: StringProperty()

class AM3D_ContextTargetItem(PropertyGroup):
    name:     StringProperty()
    obj_type: StringProperty()

class AM3D_AnimItem(PropertyGroup):
    name:         StringProperty(name="Name")
    json_payload: StringProperty(name="JSON")
    behaviors_payload: StringProperty(name="Behaviors", default="[]")
    prompt:       StringProperty(name="Prompt")
    target_name:  StringProperty(name="Target")
    context_targets_json: StringProperty(name="Context targets", default="[]")
    applied:      BoolProperty(default=False)
    anim_type:    StringProperty(default="mixed")

class AM3D_ScriptItem(PropertyGroup):
    name:        StringProperty(name="Nombre")
    description: StringProperty(name="Descripci\u00f3n")
    code:        StringProperty(name="C\u00f3digo")

# ══════════════════════════════════════════════════════
#  CACHÉ DE ESCENA  —  evita scan en cada redibujado
# ══════════════════════════════════════════════════════
_scene_cache: dict = {}          # keyed por nombre de escena
_scene_cache_time: dict = {}     # timestamps por escena
SCENE_CACHE_TTL: float = 1.0    # segundos antes de re-escanear


def get_cached_scene_context(scene):
    """
    Devuelve el contexto de escena con caché por nombre de escena.
    Refresca si pasó más de TTL segundos o cambió el número de objetos.
    FIX v0.6: antes usaba un dict global único, fallaba con múltiples escenas.
    """
    key = scene.name
    now = time.monotonic()
    obj_count = len(scene.objects)
    cached = _scene_cache.get(key)
    if (cached is None or
            now - _scene_cache_time.get(key, 0.0) > SCENE_CACHE_TTL or
            cached.get("_obj_count") != obj_count):
        ctx = scan_scene_context(scene)
        ctx["_obj_count"] = obj_count
        _scene_cache[key] = ctx
        _scene_cache_time[key] = now
    return _scene_cache[key]


def _is_animatable_object(obj):
    return obj is not None and obj.type in ANIMATABLE_OBJECT_TYPES


def _object_transform_snapshot(obj):
    return {
        "location": [round(v, 3) for v in obj.location],
        "rotation": _get_rotation_degrees(obj, digits=2),
        "scale": [round(v, 3) for v in obj.scale],
    }


def _count_animatable_objects(scene):
    return sum(1 for obj in scene.objects if _is_animatable_object(obj))


def _unique_animatable_objects(objects):
    result = []
    seen = set()
    for obj in objects or []:
        if not _is_animatable_object(obj) or obj.name in seen:
            continue
        result.append(obj)
        seen.add(obj.name)
    return result


def _context_target_meta(obj):
    return {"name": obj.name, "type": obj.type}


def _get_scene_context_target_names(scene):
    return [item.name for item in getattr(scene, "am3d_context_targets", []) if item.name]


def _resolve_context_objects(scene, target_obj=None, stored_names=None):
    names = []
    if stored_names is None:
        stored_names = _get_scene_context_target_names(scene)
    names.extend(str(name).strip() for name in stored_names if str(name).strip())
    if target_obj:
        names.insert(0, target_obj.name)

    objects = []
    seen = set()
    for name in names:
        obj = scene.objects.get(name)
        if not _is_animatable_object(obj) or obj.name in seen:
            continue
        objects.append(obj)
        seen.add(obj.name)

    if not objects and target_obj:
        return [target_obj]
    return objects


def _resolve_target_object(context, preferred_name=""):
    scene = context.scene
    target_name = (preferred_name or scene.am3d_target_name).strip()
    if target_name:
        obj = scene.objects.get(target_name)
        return obj if _is_animatable_object(obj) else None

    if _is_animatable_object(context.object):
        return context.object

    animatable_objects = [obj for obj in scene.objects if _is_animatable_object(obj)]
    if len(animatable_objects) == 1:
        return animatable_objects[0]
    return None


def _context_scope_label(context_names, target_name=""):
    names = [name for name in context_names if name]
    if len(names) > 1:
        return "el contexto seleccionado"
    if names:
        return f"'{names[0]}'"
    if target_name:
        return f"'{target_name}'"
    return "la escena"


def _target_instruction(target_obj):
    base = f"Selected animation target: '{target_obj.name}' ({target_obj.type}). "
    if target_obj.type == "ARMATURE":
        return base + "Generate keyframes only for this armature object and its bones. Do not animate other scene elements."
    if target_obj.type == "CAMERA":
        return base + "Generate keyframes only for this camera object and its camera properties. Do not animate other scene elements."
    if target_obj.type == "LIGHT":
        return base + "Generate keyframes only for this light object and its light properties. Do not animate other scene elements."
    return base + "Generate keyframes only for this object. Do not animate other scene elements."


def _context_instruction(target_obj, context_objects):
    base = f"Primary animation target: '{target_obj.name}' ({target_obj.type}). "
    extras = [obj for obj in _unique_animatable_objects(context_objects) if not target_obj or obj.name != target_obj.name]
    if not extras:
        return base + "Generate keyframes only for this selected target."
    extra_desc = ", ".join(f"{obj.name} ({obj.type})" for obj in extras)
    return (
        base
        + "This target is the main focus. "
        + f" Additional selected context objects: {extra_desc}. "
        + "You may coordinate motion, camera cuts, and property changes across any object in this selected context, but never animate objects outside it."
    )


def _target_icon(target_obj):
    if not target_obj:
        return "QUESTION"
    return {
        "ARMATURE": "ARMATURE_DATA",
        "CAMERA": "CAMERA_DATA",
        "LIGHT": "LIGHT",
    }.get(target_obj.type, "OBJECT_DATA")


def _normalize_message(message, limit=700):
    text = " ".join(str(message).split())
    if len(text) > limit:
        return text[:limit - 1] + "..."
    return text


def _append_log(scene, level, message):
    if scene is None or not hasattr(scene, "am3d_logs"):
        return
    entry = scene.am3d_logs.add()
    entry.level = level
    entry.message = _normalize_message(message)
    while len(scene.am3d_logs) > MAX_LOG_ENTRIES:
        scene.am3d_logs.remove(0)
    scene.am3d_log_index = max(0, len(scene.am3d_logs) - 1)


def _set_status(scene, level, message):
    if scene is None:
        return
    prefix = {
        "ERROR": "✗",
        "WARNING": "⚠",
        "INFO": "✓",
    }.get(level, "•")
    scene.am3d_status = f"{prefix} {_normalize_message(message, limit=180)}"


def _notify(operator, scene, level, message, update_status=True, status_message=None):
    level = str(level).upper()
    normalized = _normalize_message(message)
    _append_log(scene, level, normalized)
    if update_status:
        _set_status(scene, level, status_message or normalized)
    if operator is not None:
        operator.report({level} if level in {"ERROR", "WARNING", "INFO"} else {"INFO"}, normalized)


def _load_json_list(raw_value, field_name):
    text = (raw_value or "").strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} debe ser una lista JSON.")
    return parsed


def _load_string_list(raw_value, field_name):
    return [str(value).strip() for value in _load_json_list(raw_value, field_name) if str(value).strip()]


def _iter_actions_for_object(obj):
    if obj.animation_data and obj.animation_data.action:
        yield obj.animation_data.action
    data = getattr(obj, "data", None)
    if data and getattr(data, "animation_data", None) and data.animation_data.action:
        yield data.animation_data.action


def _set_action_interpolation(action, mode):
    changed = False
    for fcurve in action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = mode
            changed = True
    return changed


def _interpolation_matches(obj, mode):
    found = False
    for action in _iter_actions_for_object(obj):
        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                found = True
                if keyframe.interpolation != mode:
                    return False
    return True if found else None


def _resolve_data_path_owner(root, data_path):
    if not data_path:
        raise ValueError("Ruta de propiedad vacía.")

    owner = root
    parts = data_path.split(".")
    for part in parts[:-1]:
        if not hasattr(owner, part):
            raise AttributeError(f"'{type(owner).__name__}' no tiene atributo '{part}'.")
        owner = getattr(owner, part)

    leaf = parts[-1]
    if not hasattr(owner, leaf):
        raise AttributeError(f"'{type(owner).__name__}' no tiene atributo '{leaf}'.")
    return owner, leaf


def _get_rotation_degrees(target, digits=2):
    rotation_mode = getattr(target, "rotation_mode", "XYZ")

    try:
        if rotation_mode == "QUATERNION":
            euler = target.rotation_quaternion.to_euler("XYZ")
        elif rotation_mode == "AXIS_ANGLE":
            axis_angle = target.rotation_axis_angle
            euler = Quaternion((axis_angle[1], axis_angle[2], axis_angle[3]), axis_angle[0]).to_euler("XYZ")
        else:
            euler = target.rotation_euler.to_matrix().to_euler("XYZ")
    except Exception:
        euler = Euler((0.0, 0.0, 0.0), "XYZ")

    return [round(math.degrees(v), digits) for v in euler]


def _apply_rotation_from_degrees(target, rotation_values):
    euler_xyz = Euler(tuple(math.radians(float(v)) for v in rotation_values[:3]), "XYZ")
    rotation_mode = getattr(target, "rotation_mode", "XYZ")

    if rotation_mode == "QUATERNION":
        target.rotation_quaternion = euler_xyz.to_quaternion()
        return "rotation_quaternion"

    if rotation_mode == "AXIS_ANGLE":
        quat = euler_xyz.to_quaternion()
        axis = quat.axis
        target.rotation_axis_angle = (quat.angle, axis.x, axis.y, axis.z)
        return "rotation_axis_angle"

    euler_mode = rotation_mode if rotation_mode in EULER_ROTATION_MODES else "XYZ"
    target.rotation_euler = euler_xyz.to_matrix().to_euler(euler_mode)  # type: ignore[arg-type]
    return "rotation_euler"


def _normalize_compare_value(value, degrees=False, digits=4):
    if isinstance(value, dict):
        return {k: _normalize_compare_value(v, degrees=degrees, digits=digits) for k, v in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "to_list"):
        value = value.to_list()
    elif hasattr(value, "__iter__") and not isinstance(value, (int, float, bool)):
        value = list(value)

    if isinstance(value, list):
        return [_normalize_compare_value(v, degrees=degrees, digits=digits) for v in value]
    if isinstance(value, tuple):
        return [_normalize_compare_value(v, degrees=degrees, digits=digits) for v in value]
    if isinstance(value, (int, float)):
        number = float(value)
        if degrees:
            number = math.degrees(number)
        return round(number, digits)
    return value


def _coerce_property_value(current_value, new_value):
    if isinstance(current_value, bool):
        if isinstance(new_value, bool):
            return new_value
        if isinstance(new_value, (int, float)):
            return bool(new_value)
        raise TypeError("Se esperaba un booleano.")

    if isinstance(current_value, int) and not isinstance(current_value, bool):
        if isinstance(new_value, (int, float)):
            return int(new_value)
        raise TypeError("Se esperaba un número entero.")

    if isinstance(current_value, float):
        if isinstance(new_value, (int, float)):
            return float(new_value)
        raise TypeError("Se esperaba un número real.")

    if isinstance(current_value, (str, bytes)):
        if isinstance(new_value, (str, bytes)):
            return str(new_value)
        raise TypeError("Se esperaba un texto.")

    if hasattr(current_value, "to_list"):
        current_value = current_value.to_list()  # type: ignore[union-attr]
    elif hasattr(current_value, "__iter__") and not isinstance(current_value, (dict, str, bytes)):
        current_value = list(current_value)  # type: ignore[arg-type]

    if isinstance(current_value, list):
        if not isinstance(new_value, (list, tuple)):
            raise TypeError("Se esperaba una lista de valores.")
        if len(current_value) != len(new_value):
            raise ValueError(f"Se esperaban {len(current_value)} valores y llegaron {len(new_value)}.")
        return [_coerce_property_value(cur_item, new_item) for cur_item, new_item in zip(current_value, new_value)]

    if isinstance(current_value, tuple):
        if not isinstance(new_value, (list, tuple)):
            raise TypeError("Se esperaba una secuencia de valores.")
        if len(current_value) != len(new_value):
            raise ValueError(f"Se esperaban {len(current_value)} valores y llegaron {len(new_value)}.")
        return tuple(_coerce_property_value(cur_item, new_item) for cur_item, new_item in zip(current_value, new_value))

    return new_value


def _value_difference(current_value, original_value):
    left = _normalize_compare_value(current_value)
    right = _normalize_compare_value(original_value)

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf")
        return sum(_value_difference(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return sum(_value_difference(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _camera_cut_marker_name(frame, camera_name):
    return f"AM3D_CAM_{int(frame)}_{camera_name}"


def _get_active_camera_at_frame(scene, frame):
    active_marker = None
    for marker in scene.timeline_markers:
        if not getattr(marker, "camera", None):
            continue
        if marker.frame > frame:
            continue
        if active_marker is None or marker.frame >= active_marker.frame:
            active_marker = marker
    if active_marker and active_marker.camera:
        return active_marker.camera
    return scene.camera


def _camera_cut_state(scene, context_objects=None):
    allowed_cameras = {
        obj.name for obj in _unique_animatable_objects(context_objects or [])
        if obj.type == "CAMERA"
    }
    markers = []
    for marker in sorted(scene.timeline_markers, key=lambda item: item.frame):
        camera = getattr(marker, "camera", None)
        if not camera:
            continue
        if allowed_cameras and camera.name not in allowed_cameras:
            continue
        markers.append({
            "frame": int(marker.frame),
            "camera": camera.name,
            "name": marker.name,
        })
    return markers


def _collect_scene_behavior_state(scene, context_objects=None):
    state = {
        "frame_range": [scene.frame_start, scene.frame_end],
        "fps": scene.render.fps,
    }
    active_camera = _get_active_camera_at_frame(scene, scene.frame_current)
    allowed_cameras = {
        obj.name for obj in _unique_animatable_objects(context_objects or [])
        if obj.type == "CAMERA"
    }
    if active_camera and (not allowed_cameras or active_camera.name in allowed_cameras):
        state["active_camera"] = active_camera.name
    camera_cuts = _camera_cut_state(scene, context_objects=context_objects)
    if camera_cuts:
        state["camera_cuts"] = camera_cuts
    return state


def _get_action_target(action, scene, target_obj=None):
    atype = action.get("type", "bone")
    if atype == "bone":
        armature_name = str(action.get("object", "")).strip()
        if armature_name:
            armature_obj = scene.objects.get(armature_name) or bpy.data.objects.get(armature_name)
            if armature_obj and armature_obj.type == "ARMATURE":
                return armature_obj.pose.bones.get(action.get("bone", ""))
        if target_obj and target_obj.type == "ARMATURE":
            return target_obj.pose.bones.get(action.get("bone", ""))
        for obj in scene.objects:
            if obj.type == "ARMATURE" and action.get("bone", "") in obj.pose.bones:
                return obj.pose.bones.get(action.get("bone", ""))
        return None

    obj_name = action.get("object", "")
    if target_obj and target_obj.name == obj_name:
        return target_obj
    return scene.objects.get(obj_name) or bpy.data.objects.get(obj_name)


def _manual_edit_lines_for_action(action, scene, target_obj=None):
    atype = action.get("type", "bone")
    frame = int(action.get("frame", 0))
    edits = []

    try:
        target = _get_action_target(action, scene, target_obj=target_obj)
        if target is None:
            return edits

        if atype == "bone":
            bone_name = action.get("bone", "")
            channels = (
                ("loc", [round(v, 4) for v in target.location], 0.01, "posición"),
                ("rot", _get_rotation_degrees(target), 1.0, "rotación"),
                ("scale", [round(v, 4) for v in target.scale], 0.01, "escala"),
            )
            for field, current_value, tolerance, label in channels:
                if field in action and _value_difference(current_value, action[field]) > tolerance:
                    edits.append(f"Frame {frame}: hueso '{bone_name}' {label} editada")

        elif atype == "object":
            object_name = action.get("object", "")
            channels = (
                ("loc", [round(v, 4) for v in target.location], 0.01, "posición"),
                ("rot", _get_rotation_degrees(target), 1.0, "rotación"),
                ("scale", [round(v, 4) for v in target.scale], 0.01, "escala"),
            )
            for field, current_value, tolerance, label in channels:
                if field in action and _value_difference(current_value, action[field]) > tolerance:
                    edits.append(f"Frame {frame}: objeto '{object_name}' {label} editada")

        elif atype == "property":
            object_name = action.get("object", "")
            prop_path = action.get("property", "")
            owner, leaf = _resolve_data_path_owner(target, prop_path)
            current_value = _normalize_compare_value(getattr(owner, leaf))
            if _value_difference(current_value, action.get("value")) > 0.01:
                edits.append(f"Frame {frame}: propiedad '{prop_path}' de '{object_name}' editada")

    except Exception:
        return edits

    return edits

# ══════════════════════════════════════════════════════
#  HTTP  —  retries + backoff exponencial + cancelación
# ══════════════════════════════════════════════════════
def http_post_with_retries(url, data_bytes, headers,
                           timeout=DEFAULT_TIMEOUT,
                           retries=DEFAULT_RETRIES,
                           backoff=DEFAULT_BACKOFF,
                           cancel_event: Optional[threading.Event] = None):
    """
    POST con reintentos, backoff exponencial y cancelación real.

    FIX v0.6: urlopen bloqueante no puede interrumpirse desde otro hilo.
    Solución: usamos socket raw con timeout corto (SOCKET_TIMEOUT) y leemos
    la respuesta en chunks, verificando cancel_event en cada chunk.
    Esto permite cancelar una generación larga en ~SOCKET_TIMEOUT segundos.
    """
    import socket

    last_err = None
    attempt = 0
    unlimited_retries = retries < 0

    while True:
        if cancel_event and cancel_event.is_set():
            return None, "Petición cancelada por el usuario."
        try:
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=SOCKET_TIMEOUT) as resp:
                # Leer en chunks para poder cancelar durante la transferencia
                chunks = []
                deadline = time.monotonic() + timeout
                while True:
                    if cancel_event and cancel_event.is_set():
                        return None, "Petición cancelada por el usuario."
                    if time.monotonic() > deadline:
                        return None, f"Timeout total de {timeout}s superado."
                    try:
                        chunk = resp.read(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks).decode("utf-8"), None

        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.reason}"
            if 400 <= e.code < 500:
                return None, last_err
        except urllib.error.URLError as e:
            last_err = f"URL error: {e.reason}"
        except Exception as e:
            last_err = str(e)

        attempt += 1
        if not unlimited_retries and attempt > retries:
            break

        wait = min(backoff * (2 ** min(attempt - 1, 8)), MAX_BACKOFF_WAIT)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if cancel_event and cancel_event.is_set():
                return None, "Petición cancelada por el usuario."
            time.sleep(0.1)

    return None, last_err or "No se pudo contactar a LM Studio."


def call_lm_studio(messages, url, model, temperature, timeout,
                   retries=DEFAULT_RETRIES,
                   cancel_event: Optional[threading.Event] = None):
    """Llama a LM Studio (OpenAI-compatible). Devuelve (content_str, error_str)."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    body, err = http_post_with_retries(
        url,
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
        timeout=timeout,
        retries=retries,
        cancel_event=cancel_event,
    )
    if err:
        return None, err
    if body is None:
        return None, "Respuesta vacía inesperada del servidor."
    # Parsear SSE (Server-Sent Events) del stream
    content_parts = []
    for line in body.splitlines():
        line = line.strip()
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                content_parts.append(piece)
        except (json.JSONDecodeError, KeyError):
            continue
    content = "".join(content_parts).strip()
    if not content:
        return None, f"LM Studio devolvió contenido vacío. El modelo puede estar solo razonando sin generar JSON. Raw: {body[:300]}"
    return content, None

# ══════════════════════════════════════════════════════
#  PREPARSER  —  extrae JSON aunque venga con basura
# ══════════════════════════════════════════════════════
def extract_json_block(text):
    """
    Localiza el primer objeto/array JSON válido en el texto, tolerando
    texto basura alrededor, fences de markdown y comas trailing.
    Devuelve el string JSON limpio o None.
    """
    if not text:
        return None
    s = text.strip().replace("```json", "").replace("```", "").strip()

    start = next((i for i, c in enumerate(s) if c in "{["), None)
    if start is None:
        return None

    open_ch  = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, end = 0, None
    in_string, escape = False, False

    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        return None

    candidate = s[start:end + 1]
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        import re as _re
        # Solo reemplaza comas trailing fuera de strings (antes de ] o })
        fixed = _re.sub(r',\s*([}\]])', r'\1', candidate)
        try:
            json.loads(fixed)
            return fixed
        except Exception:
            return None


def parse_animation_json(text):
    """
    Normaliza la respuesta de la IA al formato interno.
    Devuelve (result_dict, error_str).
    """
    block = extract_json_block(text)
    if not block:
        return None, f"No se pudo extraer JSON.\nTexto recibido:\n{(text or '')[:400]}"

    try:
        data = json.loads(block)
    except Exception as e:
        return None, f"JSON inválido: {e}"

    result = {"animations": [], "behaviors": [], "notes": ""}

    if isinstance(data, list):
        result["animations"] = data
    elif isinstance(data, dict):
        for key in ("animations", "keyframes", "actions", "frames"):
            if key in data and isinstance(data[key], list):
                result["animations"] = data[key]
                break
        behaviors = data.get("behaviors", [])
        if behaviors and not isinstance(behaviors, list):
            return None, "El campo 'behaviors' debe ser una lista."
        result["behaviors"] = behaviors
        result["notes"]     = data.get("notes", "")
    else:
        return None, f"Formato JSON no soportado: {type(data).__name__}"

    if not result["animations"] and not result["behaviors"]:
        return None, "La IA no generó acciones ni behaviors válidos. Revisa el prompt o el modelo."

    return result, None

# ══════════════════════════════════════════════════════
#  ESCENA  —  scan, validate, apply
# ══════════════════════════════════════════════════════
def scan_scene_context(scene, target_obj=None, context_objects=None):
    """Devuelve un dict con los elementos animables del contexto seleccionado solamente.
    Si context_objects es None y target_obj es None, usa solo el objeto activo o nada.
    NUNCA hace un scan completo de la escena por defecto.
    """
    ctx = {
        "armatures": [],
        "objects":   [],
        "cameras":   [],
        "lights":    [],
        "frame_range": [scene.frame_start, scene.frame_end],
        "fps": scene.render.fps,
    }
    # Solo objetos explícitamente seleccionados — nunca toda la escena
    objects = _unique_animatable_objects(context_objects or ([target_obj] if target_obj else []))
    if target_obj:
        ctx["selected_target"] = {"name": target_obj.name, "type": target_obj.type}
    if objects:
        ctx["context_targets"] = [_context_target_meta(obj) for obj in objects]

    active_camera = _get_active_camera_at_frame(scene, scene.frame_current)
    if active_camera:
        ctx["active_camera"] = active_camera.name
    camera_cuts = _camera_cut_state(scene, context_objects=objects)
    if camera_cuts:
        ctx["camera_cuts"] = camera_cuts

    for obj in objects:
        d = {"name": obj.name, "type": obj.type}
        if obj.type == "ARMATURE":
            d["bones"] = [b.name for b in obj.pose.bones]
            ctx["armatures"].append(d)
        elif obj.type == "CAMERA":
            d.update(_object_transform_snapshot(obj))
            d["properties"] = {"lens": round(obj.data.lens, 2)}
            ctx["cameras"].append(d)
        elif obj.type == "LIGHT":
            d.update(_object_transform_snapshot(obj))
            d["properties"] = {"energy": round(obj.data.energy, 2), "type": obj.data.type}
            ctx["lights"].append(d)
        elif obj.type in ("MESH", "CURVE", "EMPTY"):
            d.update(_object_transform_snapshot(obj))
            ctx["objects"].append(d)
    return ctx


def _target_constraints(scene_context):
    selected = scene_context.get("selected_target", {})
    target_name = selected.get("name", "")
    target_type = selected.get("type", "")
    context_names = [entry.get("name", "") for entry in scene_context.get("context_targets", []) if entry.get("name")]

    armatures = {
        arm["name"]: set(arm.get("bones", []))
        for arm in scene_context.get("armatures", [])
    }
    object_names = set(armatures.keys())
    for key in ("objects", "cameras", "lights"):
        object_names.update(o["name"] for o in scene_context.get(key, []))

    allowed_objects = object_names
    allowed_bones = set().union(*armatures.values()) if armatures else set()

    return target_name, target_type, allowed_objects, allowed_bones, armatures, context_names


def validate_actions(actions, scene_context):
    """Valida y filtra keyframes. Devuelve (valid_list, warnings_list)."""
    valid, warnings = [], []
    target_name, _, allowed_objects, _, armatures, context_names = _target_constraints(scene_context)
    scope_label = _context_scope_label(context_names, target_name=target_name)

    for i, a in enumerate(actions):
        if not isinstance(a, dict):
            warnings.append(f"Acción {i}: no es dict.")
            continue

        atype = a.get("type", "bone")

        if atype == "bone":
            bone = a.get("bone")
            if not bone:
                warnings.append(f"Acción {i}: sin campo 'bone'.")
                continue
            armature_name = str(a.get("object", "")).strip()
            if armature_name:
                if armature_name not in armatures:
                    warnings.append(f"Acción {i}: armature '{armature_name}' fuera de {scope_label}.")
                    continue
                if bone not in armatures[armature_name]:
                    warnings.append(f"Acción {i}: hueso '{bone}' no pertenece al armature '{armature_name}'.")
                    continue
            else:
                owners = [name for name, bones in armatures.items() if bone in bones]
                if not owners:
                    warnings.append(f"Acción {i}: hueso '{bone}' no existe en {scope_label}.")
                    continue
                if len(owners) > 1:
                    owners_str = ", ".join(owners)
                    warnings.append(f"Acción {i}: hueso '{bone}' es ambiguo entre {owners_str}. Incluye el campo 'object'.")
                    continue

        elif atype in ("object", "property"):
            obj_name = a.get("object")
            if not obj_name:
                warnings.append(f"Acción {i}: sin campo 'object'.")
                continue
            if obj_name not in allowed_objects:
                warnings.append(f"Acción {i}: objeto '{obj_name}' fuera de {scope_label}.")
                continue
        else:
            warnings.append(f"Acción {i}: tipo '{atype}' desconocido.")
            continue

        frame = a.get("frame")
        if frame is None or not isinstance(frame, (int, float)):
            warnings.append(f"Acción {i}: frame inválido.")
            continue
        if int(frame) < 1:
            warnings.append(f"Acción {i}: frame debe ser >= 1.")
            continue

        valid.append(a)

    return valid, warnings


def validate_behaviors(behaviors, scene_context):
    """Valida behaviors soportados. Devuelve (valid_list, warnings_list)."""
    valid, warnings = [], []
    target_name, _, allowed_objects, _, _, context_names = _target_constraints(scene_context)
    scope_label = _context_scope_label(context_names, target_name=target_name)
    camera_names = {camera["name"] for camera in scene_context.get("cameras", [])}

    for i, behavior in enumerate(behaviors):
        if not isinstance(behavior, dict):
            warnings.append(f"Behavior {i}: no es dict.")
            continue

        btype = behavior.get("type", "")

        if btype == "set_frame_range":
            start = behavior.get("start")
            end = behavior.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                warnings.append(f"Behavior {i}: frame range inválido.")
                continue
            start = int(start)
            end = int(end)
            if start < 1 or end < start:
                warnings.append(f"Behavior {i}: start/end inválidos ({start}, {end}).")
                continue
            valid.append({"type": "set_frame_range", "start": start, "end": end})

        elif btype == "set_scene_fps":
            fps = behavior.get("fps")
            if not isinstance(fps, (int, float)):
                warnings.append(f"Behavior {i}: fps inválido.")
                continue
            fps = int(fps)
            if fps < 1 or fps > 240:
                warnings.append(f"Behavior {i}: fps fuera de rango ({fps}).")
                continue
            valid.append({"type": "set_scene_fps", "fps": fps})

        elif btype == "set_interpolation":
            mode = str(behavior.get("mode", "")).upper().strip()
            obj_name = str(behavior.get("object") or target_name or "*").strip()
            if mode not in SUPPORTED_INTERPOLATION_MODES:
                warnings.append(f"Behavior {i}: interpolación '{mode}' no soportada.")
                continue
            if obj_name not in {"*", "all"}:
                if obj_name not in allowed_objects:
                    warnings.append(f"Behavior {i}: objeto '{obj_name}' fuera de {scope_label}.")
                    continue
            valid.append({"type": "set_interpolation", "mode": mode, "object": obj_name})

        elif btype in {"set_active_camera", "camera_cut"}:
            frame = behavior.get("frame")
            camera_name = str(behavior.get("object") or behavior.get("camera") or "").strip()
            if not isinstance(frame, (int, float)) or int(frame) < 1:
                warnings.append(f"Behavior {i}: frame inválido para cambio de cámara.")
                continue
            if not camera_name:
                warnings.append(f"Behavior {i}: falta la cámara destino.")
                continue
            if camera_name not in allowed_objects:
                warnings.append(f"Behavior {i}: cámara '{camera_name}' fuera de {scope_label}.")
                continue
            if camera_name not in camera_names:
                warnings.append(f"Behavior {i}: '{camera_name}' no es una cámara válida.")
                continue
            valid.append({"type": "set_active_camera", "frame": int(frame), "object": camera_name})

        else:
            warnings.append(f"Behavior {i}: tipo '{btype}' no soportado.")

    return valid, warnings


def apply_actions(obj, actions):
    """
    Aplica keyframes al armature/objeto activo.
    Para acciones de tipo 'object' y 'property' accede a bpy.data.objects directamente.
    Devuelve lista de advertencias.

    CORRECCIÓN v0.5: restauración de modo corregida usando comparación directa
    del modo actual en lugar del modo previo registrado.
    """
    scene = bpy.context.scene
    prev_frame = scene.frame_current
    warnings = []
    pose_modes = {}

    def ensure_pose_mode(armature_obj):
        if not armature_obj or armature_obj.type != "ARMATURE":
            return
        if armature_obj.name not in pose_modes:
            pose_modes[armature_obj.name] = armature_obj.mode
        bpy.context.view_layer.objects.active = armature_obj
        if armature_obj.mode != "POSE":
            bpy.ops.object.mode_set(mode="POSE")

    for a in actions:
        frame  = int(a["frame"])
        atype  = a.get("type", "bone")
        scene.frame_set(frame)

        try:
            if atype == "bone":
                armature_name = str(a.get("object", "")).strip()
                if armature_name:
                    bone_obj = scene.objects.get(armature_name) or bpy.data.objects.get(armature_name)
                else:
                    bone_obj = obj if obj and obj.type == "ARMATURE" else None
                    if bone_obj is None:
                        matches = [candidate for candidate in scene.objects if candidate.type == "ARMATURE" and a.get("bone", "") in candidate.pose.bones]
                        if len(matches) == 1:
                            bone_obj = matches[0]
                        elif len(matches) > 1:
                            warnings.append(f"Hueso '{a.get('bone')}' ambiguo entre varios armatures (frame {frame}).")
                            continue

                if not bone_obj or bone_obj.type != "ARMATURE":
                    warnings.append(f"Armature '{armature_name or (obj.name if obj else '')}' no encontrado (frame {frame}).")
                    continue

                ensure_pose_mode(bone_obj)
                pbone = bone_obj.pose.bones.get(a["bone"])
                if not pbone:
                    warnings.append(f"Hueso '{a.get('bone')}' no encontrado (frame {frame}).")
                    continue
                if "loc" in a:
                    pbone.location = tuple(float(v) for v in a["loc"][:3])
                    pbone.keyframe_insert(data_path="location", frame=frame)
                if "rot" in a:
                    data_path = _apply_rotation_from_degrees(pbone, a["rot"])
                    pbone.keyframe_insert(data_path=data_path, frame=frame)
                if "scale" in a:
                    pbone.scale = tuple(float(v) for v in a["scale"][:3])
                    pbone.keyframe_insert(data_path="scale", frame=frame)

            elif atype == "object":
                tobj = bpy.data.objects.get(a["object"])
                if not tobj:
                    warnings.append(f"Objeto '{a['object']}' no encontrado (frame {frame}).")
                    continue
                if "loc" in a:
                    tobj.location = tuple(float(v) for v in a["loc"][:3])
                    tobj.keyframe_insert(data_path="location", frame=frame)
                if "rot" in a:
                    data_path = _apply_rotation_from_degrees(tobj, a["rot"])
                    tobj.keyframe_insert(data_path=data_path, frame=frame)
                if "scale" in a:
                    tobj.scale = tuple(float(v) for v in a["scale"][:3])
                    tobj.keyframe_insert(data_path="scale", frame=frame)

            elif atype == "property":
                tobj = bpy.data.objects.get(a["object"])
                if not tobj:
                    warnings.append(f"Objeto '{a['object']}' no encontrado para propiedad.")
                    continue
                prop_path = a.get("property", "")
                value     = a.get("value")
                if not prop_path or value is None:
                    warnings.append(f"Propiedad incompleta para '{a['object']}'.")
                    continue
                try:
                    obj_ref, leaf = _resolve_data_path_owner(tobj, prop_path)
                except Exception as e:
                    warnings.append(f"Propiedad '{prop_path}' inválida en '{a['object']}': {e}")
                    continue

                try:
                    current_value = getattr(obj_ref, leaf)
                    safe_value = _coerce_property_value(current_value, value)
                    setattr(obj_ref, leaf, safe_value)
                except Exception as e:
                    warnings.append(f"Valor inválido para '{a['object']}.{prop_path}': {e}")
                    continue

                try:
                    tobj.keyframe_insert(data_path=prop_path, frame=frame)
                except Exception as e:
                    warnings.append(f"No se pudo insertar keyframe en '{a['object']}.{prop_path}': {e}")
                    continue

        except Exception as e:
            warnings.append(f"Error frame {frame} ({atype}): {e}")

    scene.frame_set(prev_frame)

    for armature_name, prev_mode in pose_modes.items():
        armature_obj = bpy.data.objects.get(armature_name)
        if not armature_obj or armature_obj.type != "ARMATURE" or armature_obj.mode == prev_mode:
            continue
        try:
            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode=prev_mode)
        except Exception:
            pass

    return warnings


def apply_behaviors(scene, target_obj, behaviors, context_objects=None):
    """Aplica behaviors soportados y devuelve lista de advertencias."""
    warnings = []
    context_objects = _unique_animatable_objects(context_objects or ([] if target_obj is None else [target_obj]))

    for behavior in behaviors:
        btype = behavior.get("type", "")
        try:
            if btype == "set_frame_range":
                scene.frame_start = int(behavior["start"])
                scene.frame_end = int(behavior["end"])

            elif btype == "set_scene_fps":
                scene.render.fps = int(behavior["fps"])

            elif btype == "set_interpolation":
                obj_name = behavior.get("object") or (target_obj.name if target_obj else "*")
                if obj_name in {"*", "all"}:
                    objects = context_objects or [obj for obj in scene.objects if _is_animatable_object(obj)]
                else:
                    resolved = scene.objects.get(obj_name) or bpy.data.objects.get(obj_name)
                    if not resolved:
                        warnings.append(f"Behavior interpolation: objeto '{obj_name}' no encontrado.")
                        continue
                    objects = [resolved]

                changed = False
                for obj in objects:
                    for action in _iter_actions_for_object(obj):
                        changed = _set_action_interpolation(action, behavior["mode"]) or changed
                if not changed:
                    warnings.append(f"Behavior interpolation: no hay fcurves para '{obj_name}'.")

            elif btype == "set_active_camera":
                frame = int(behavior["frame"])
                camera_name = behavior.get("object") or behavior.get("camera")
                camera_obj = scene.objects.get(camera_name) or bpy.data.objects.get(camera_name)
                if not camera_obj or camera_obj.type != "CAMERA":
                    warnings.append(f"Behavior cámara: '{camera_name}' no es una cámara válida.")
                    continue

                for marker in list(scene.timeline_markers):
                    if marker.frame == frame and marker.name.startswith("AM3D_CAM_"):
                        scene.timeline_markers.remove(marker)

                marker_name = _camera_cut_marker_name(frame, camera_obj.name)
                marker = scene.timeline_markers.get(marker_name)
                if marker is None:
                    marker = scene.timeline_markers.new(marker_name, frame=frame)
                else:
                    marker.frame = frame
                marker.camera = camera_obj

                if scene.camera is None or frame <= scene.frame_start:
                    scene.camera = camera_obj

        except Exception as e:
            warnings.append(f"Behavior '{btype}': {e}")

    return warnings

# ══════════════════════════════════════════════════════
#  RIG MAPPING
# ══════════════════════════════════════════════════════
def apply_rig_mapping(actions, mapping):
    """
    Renombra huesos/objetos según el dict de mapping {source: target}.
    No muta la lista original.
    """
    if not mapping:
        return actions
    out = []
    for a in actions:
        na = dict(a)
        if a.get("type", "bone") == "bone" and "bone" in a:
            na["bone"] = mapping.get(a["bone"], a["bone"])
        if "object" in a:
            na["object"] = mapping.get(a["object"], a["object"])
        out.append(na)
    return out


def _collect_feedback_property_paths(actions, target_obj=None, context_objects=None):
    paths = []
    target_names = {obj.name for obj in _unique_animatable_objects(context_objects or [])}
    if target_obj:
        target_names.add(target_obj.name)

    if target_obj:
        if target_obj.type == "CAMERA":
            paths.append("data.lens")
        elif target_obj.type == "LIGHT":
            paths.append("data.energy")

    for action in actions:
        if action.get("type") != "property":
            continue
        if target_names and action.get("object") not in target_names:
            continue
        prop_path = str(action.get("property", "")).strip()
        if prop_path:
            paths.append(prop_path)

    return list(dict.fromkeys(paths))

# ══════════════════════════════════════════════════════
#  FEEDBACK  —  sampleo y diff manual
# ══════════════════════════════════════════════════════
def read_keyframes_from_fcurves(scene, target_obj=None, context_objects=None):
    """
    Lee solo los keyframes reales desde las F-Curves (ignora frames interpolados).
    Produce mucho menos tokens que sample_scene_animation para usar como contexto AI.
    """
    objects = _unique_animatable_objects(
        context_objects or ([target_obj] if target_obj else [])
    )

    # Recopilar todos los frame numbers únicos con keyframe real
    keyframe_frames: set[int] = set()
    for obj in objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for kp in fcurve.keyframe_points:
                    keyframe_frames.add(int(round(kp.co[0])))

    if not keyframe_frames:
        return []

    prev = scene.frame_current
    result = []
    allowed_camera_names = {obj.name for obj in objects if obj.type == "CAMERA"}

    for f in sorted(keyframe_frames):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        fd: dict[str, Any] = {"frame": f}

        active_camera = _get_active_camera_at_frame(scene, f)
        if active_camera and (not allowed_camera_names or active_camera.name in allowed_camera_names):
            fd["active_camera"] = active_camera.name

        bones_data = {}
        for obj in objects:
            if obj.type != "ARMATURE":
                continue
            for b in obj.pose.bones:
                loc = [round(v, 4) for v in b.location]
                rot = _get_rotation_degrees(b)
                scl = [round(v, 4) for v in b.scale]
                if (any(abs(v) > 1e-4 for v in loc) or
                        any(abs(v) > 0.01 for v in rot) or
                        any(abs(v - 1.0) > 1e-4 for v in scl)):
                    bones_data[f"{obj.name}:{b.name}"] = {"loc": loc, "rot": rot, "scale": scl}
        if bones_data:
            fd["bones"] = bones_data

        objs_data = {}
        for obj in objects:
            if obj.type not in ("ARMATURE", "MESH", "CAMERA", "LIGHT", "EMPTY", "CURVE"):
                continue
            loc = [round(v, 4) for v in obj.location]
            rot = _get_rotation_degrees(obj)
            scale = [round(v, 4) for v in obj.scale]
            has_transform = (
                any(abs(v) > 1e-4 for v in loc) or
                any(abs(v) > 0.01 for v in rot) or
                any(abs(v - 1.0) > 1e-4 for v in scale)
            )
            if has_transform:
                objs_data[obj.name] = {"loc": loc, "rot": rot, "scale": scale}
        if objs_data:
            fd["objects"] = objs_data

        if len(fd) > 1:
            result.append(fd)

    scene.frame_set(prev)
    return result


def sample_scene_animation(scene, start, end, step=2, target_obj=None, tracked_property_paths=None, context_objects=None):
    """
    Samplea la animación actual (bones + objetos) cada `step` frames.
    Omite huesos/objetos en reposo para mantener el JSON pequeño.
    """
    prev = scene.frame_current
    result = []
    tracked_property_paths = tracked_property_paths or []
    objects = _unique_animatable_objects(context_objects or ([target_obj] if target_obj else scene.objects))
    allowed_camera_names = {obj.name for obj in objects if obj.type == "CAMERA"}

    for f in range(start, end + 1, step):
        scene.frame_set(f)
        bpy.context.view_layer.update()  # FIX v0.6: fuerza drivers y constraints
        fd: dict[str, Any] = {"frame": f}

        active_camera = _get_active_camera_at_frame(scene, f)
        if active_camera and (not allowed_camera_names or active_camera.name in allowed_camera_names):
            fd["active_camera"] = active_camera.name

        bones_data = {}
        for obj in objects:
            if obj.type != "ARMATURE":
                continue
            for b in obj.pose.bones:
                loc = [round(v, 4) for v in b.location]
                rot = _get_rotation_degrees(b)
                scl = [round(v, 4) for v in b.scale]
                if (any(abs(v) > 1e-4 for v in loc) or
                        any(abs(v) > 0.01 for v in rot) or
                        any(abs(v - 1.0) > 1e-4 for v in scl)):
                    bones_data[f"{obj.name}:{b.name}"] = {"loc": loc, "rot": rot, "scale": scl}
        if bones_data:
            fd["bones"] = bones_data

        objs_data = {}
        for obj in objects:
            if obj.type not in ("ARMATURE", "MESH", "CAMERA", "LIGHT", "EMPTY", "CURVE"):
                continue
            loc = [round(v, 4) for v in obj.location]
            rot = _get_rotation_degrees(obj)
            scale = [round(v, 4) for v in obj.scale]
            properties = {}
            for prop_path in tracked_property_paths:
                try:
                    owner, leaf = _resolve_data_path_owner(obj, prop_path)
                    properties[prop_path] = _normalize_compare_value(getattr(owner, leaf))
                except Exception:
                    continue

            has_transform = (
                any(abs(v) > 1e-4 for v in loc) or
                any(abs(v) > 0.01 for v in rot) or
                any(abs(v - 1.0) > 1e-4 for v in scale)
            )
            if has_transform or properties:
                entry = {"loc": loc, "rot": rot, "scale": scale}
                if properties:
                    entry["properties"] = properties
                objs_data[obj.name] = entry
        if objs_data:
            fd["objects"] = objs_data

        if len(fd) > 1:
            result.append(fd)

    scene.frame_set(prev)
    return result


def detect_manual_edits(original_actions, scene, start, end, step=2, target_obj=None):
    """
    Compara keyframes originales con el estado actual de la escena.
    Devuelve lista de strings describiendo cambios detectados.
    """
    actions_by_frame = {}
    for action in original_actions:
        frame = int(action.get("frame", 0))
        if frame < start or frame > end:
            continue
        actions_by_frame.setdefault(frame, []).append(action)

    edits  = []
    prev   = scene.frame_current
    sampled_frames = set(range(start, end + 1, step))
    relevant_frames = sorted(sampled_frames.union(actions_by_frame))

    for f in relevant_frames:
        scene.frame_set(f)
        bpy.context.view_layer.update()  # FIX v0.6: fuerza drivers y constraints
        for action in actions_by_frame.get(f, []):
            edits.extend(_manual_edit_lines_for_action(action, scene, target_obj=target_obj))

    scene.frame_set(prev)
    return list(dict.fromkeys(edits))


def detect_behavior_edits(original_behaviors, scene, context_objects=None):
    edits = []
    context_lookup = {
        obj.name: obj
        for obj in _unique_animatable_objects(context_objects or scene.objects)
    }

    for behavior in original_behaviors:
        btype = behavior.get("type", "")
        try:
            if btype == "set_frame_range":
                expected = [int(behavior.get("start", scene.frame_start)), int(behavior.get("end", scene.frame_end))]
                current = [scene.frame_start, scene.frame_end]
                if current != expected:
                    edits.append(f"Behavior: frame range editado a {current[0]}-{current[1]} (antes {expected[0]}-{expected[1]})")

            elif btype == "set_scene_fps":
                expected_fps = int(behavior.get("fps", scene.render.fps))
                if scene.render.fps != expected_fps:
                    edits.append(f"Behavior: FPS editado a {scene.render.fps} (antes {expected_fps})")

            elif btype == "set_active_camera":
                frame = int(behavior.get("frame", 0))
                expected_camera = str(behavior.get("object") or behavior.get("camera") or "").strip()
                current_camera = _get_active_camera_at_frame(scene, frame)
                current_name = current_camera.name if current_camera else ""
                if current_name != expected_camera:
                    edits.append(f"Frame {frame}: cámara activa editada a '{current_name or 'ninguna'}' (antes '{expected_camera}')")

            elif btype == "set_interpolation":
                mode = str(behavior.get("mode", "")).upper().strip()
                obj_name = str(behavior.get("object") or "*").strip() or "*"
                if obj_name in {"*", "all"}:
                    objects = list(context_lookup.values())
                else:
                    resolved = context_lookup.get(obj_name) or scene.objects.get(obj_name) or bpy.data.objects.get(obj_name)
                    objects = [resolved] if (resolved is not None and _is_animatable_object(resolved)) else []

                mismatched = []
                for obj in objects:
                    if obj is None:
                        continue
                    match = _interpolation_matches(obj, mode)
                    if match is False:
                        mismatched.append(obj.name)
                if mismatched:
                    edits.append(f"Behavior: interpolación ya no coincide en {', '.join(mismatched)}")

        except Exception:
            continue

    return list(dict.fromkeys(edits))

# ── Caché del contador SFT ────────────────────────────
_sft_count_cache: int   = -1   # -1 = no inicializado
_sft_count_mtime: float = 0.0  # última mtime del archivo conocida


def _get_sft_count() -> str:
    """
    Devuelve el número de líneas del archivo SFT sin leer el archivo completo en cada frame.
    FIX v0.6: antes se abría el archivo en cada redibujado del panel.
    Ahora solo re-cuenta si el mtime cambió (es decir, se guardó un nuevo par).
    """
    global _sft_count_cache, _sft_count_mtime
    try:
        mtime = os.path.getmtime(SFT_FILENAME)
        if mtime != _sft_count_mtime:
            with open(SFT_FILENAME, "rb") as f:
                _sft_count_cache = sum(1 for _ in f)
            _sft_count_mtime = mtime
        return str(_sft_count_cache)
    except Exception:
        return "?"


# ══════════════════════════════════════════════════════
#  SFT DATASET EXPORT
# ══════════════════════════════════════════════════════
def save_sft_pair(prompt, scene_context, generated, corrected):
    """
    Guarda un par {generado, corregido} en JSONL para fine-tuning futuro.
    Devuelve (bool, error_str).
    """
    record = {
        "timestamp":     time.time(),
        "prompt":        prompt,
        "scene_context": scene_context,
        "generated":     generated,
        "corrected":     corrected,
    }
    try:
        with open(SFT_FILENAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True, None
    except Exception as e:
        return False, str(e)

# ══════════════════════════════════════════════════════
#  THREADING  —  con soporte de cancelación
# ══════════════════════════════════════════════════════
class _RequestThread(threading.Thread):
    def __init__(self, messages, url, model, temperature, timeout, retries):
        super().__init__(daemon=True)
        self.messages      = messages
        self.url           = url
        self.model         = model
        self.temperature   = temperature
        self.timeout       = timeout
        self.retries       = retries
        self.result_content = None
        self.result_error   = None
        self.cancel_event   = threading.Event()   # ← NUEVO: señal de cancelación

    def cancel(self):
        """Solicita cancelación del hilo en el próximo check."""
        self.cancel_event.set()

    def run(self):
        self.result_content, self.result_error = call_lm_studio(
            self.messages, self.url, self.model,
            self.temperature, self.timeout,
            retries=self.retries,
            cancel_event=self.cancel_event,
        )


_active_thread: Optional[_RequestThread] = None
_pending_data:  dict                     = {}


def _build_messages(sc, new_user_content):
    """
    Historial de conversación + nuevo mensaje.
    FIX v0.6: limita a MAX_CONV_TURNS últimos turnos para evitar prompts gigantes.
    Cada turno = 1 mensaje user + 1 assistant, por eso usamos MAX_CONV_TURNS * 2.
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    history = list(sc.am3d_conversation)
    max_msgs = MAX_CONV_TURNS * 2
    if len(history) > max_msgs:
        history = history[-max_msgs:]   # últimos N turnos
    for m in history:
        msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": new_user_content})
    return msgs


def _get_rig_mapping(sc):
    try:
        return json.loads(sc.am3d_rigmap_json) if sc.am3d_rigmap_json.strip() else {}
    except Exception:
        return {}

# ══════════════════════════════════════════════════════
#  OPERADORES
# ══════════════════════════════════════════════════════

# ── Helper compartido para finalizar petición ──────────
def _finish_request(self, context, raw_content):
    """
    Parsea la respuesta de la IA, valida, aplica mapping y guarda en la lista.
    Devuelve {'FINISHED'} o {'CANCELLED'}.
    """
    global _pending_data

    sc = context.scene

    parsed, err = parse_animation_json(raw_content)
    if err or parsed is None:
        _notify(self, sc, "ERROR", f"Error parse: {err or 'resultado vacío'}")
        return {"CANCELLED"}

    scene_ctx  = _pending_data.get("scene_context", {})
    animations = parsed.get("animations", [])
    behaviors  = parsed.get("behaviors", [])
    notes      = parsed.get("notes", "")
    target_name = _pending_data.get("target_name", "")
    context_target_names = _pending_data.get("context_target_names", [])

    mapping = _get_rig_mapping(sc)
    mapped_actions = apply_rig_mapping(animations, mapping)
    mapped_behaviors = apply_rig_mapping(behaviors, mapping)

    valid_actions, action_warnings = validate_actions(mapped_actions, scene_ctx)
    valid_behaviors, behavior_warnings = validate_behaviors(mapped_behaviors, scene_ctx)
    for warning in action_warnings + behavior_warnings:
        _notify(self, sc, "WARNING", warning, update_status=False)

    if not valid_actions and not valid_behaviors:
        target_label = target_name or "seleccionado"
        _notify(self, sc, "ERROR", f"Sin acciones ni behaviors válidos para '{target_label}'.")
        return {"CANCELLED"}

    item              = sc.animotion3d_animations.add()
    item.name         = _pending_data.get("item_name", "AI anim")[:50]
    item.prompt       = _pending_data.get("prompt", "")
    item.target_name  = target_name
    item.context_targets_json = json.dumps(context_target_names)
    item.json_payload = json.dumps(valid_actions)
    item.behaviors_payload = json.dumps(valid_behaviors)
    if valid_actions and valid_behaviors:
        item.anim_type = "mixed"
    elif valid_behaviors:
        item.anim_type = "behavior"
    else:
        item.anim_type = "animation"

    sc.animotion3d_index = len(sc.animotion3d_animations) - 1

    mu         = sc.am3d_conversation.add()
    mu.role    = "user"
    mu.content = _pending_data.get("full_prompt", "")
    ma         = sc.am3d_conversation.add()
    ma.role    = "assistant"
    ma.content = raw_content

    summary = []
    if valid_actions:
        summary.append(f"{len(valid_actions)} keyframes")
    if valid_behaviors:
        summary.append(f"{len(valid_behaviors)} behaviors")
    target_str = f" [{target_name}]" if target_name else ""
    note_str = f" | {notes}" if notes else ""
    status_msg = f"Generado {', '.join(summary)}{target_str}.{note_str}"
    _notify(
        self,
        sc,
        "INFO",
        f"Animación '{item.name}' lista con {', '.join(summary)}. Warnings: {len(action_warnings) + len(behavior_warnings)}",
        status_message=status_msg,
    )
    return {"FINISHED"}


# ── Generate ──────────────────────────────────────────
class AM3D_OT_GenerateAnimation(Operator):
    bl_idname     = "am3d.generate_animation"
    bl_label      = "Generar con IA"
    bl_description = "Envía el prompt a LM Studio y genera una animación"

    _timer = None

    def modal(self, context, event):
        global _active_thread
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if not (_active_thread and _active_thread.is_alive()):
            self.cancel(context)
            if _active_thread is None:
                return {"CANCELLED"}

            content = _active_thread.result_content
            error   = _active_thread.result_error
            _active_thread = None

            if error:
                level = "INFO" if "cancelada" in error.lower() else "ERROR"
                _notify(self, context.scene, level, error)
                return {"CANCELLED"}

            return _finish_request(self, context, content)

        context.scene.am3d_status = "⏳ Esperando a LM Studio…"
        return {"PASS_THROUGH"}

    def execute(self, context):
        global _active_thread, _pending_data

        sc = context.scene

        if _active_thread and _active_thread.is_alive():
            _notify(self, sc, "WARNING", "Petición en curso, espera.", update_status=False)
            return {"CANCELLED"}

        prompt = sc.am3d_prompt_text.strip()
        if not prompt:
            _notify(self, sc, "ERROR", "Escribe un prompt primero.")
            return {"CANCELLED"}

        target_obj = _resolve_target_object(context)
        if target_obj is None:
            if _count_animatable_objects(sc) > 1:
                _notify(self, sc, "ERROR", "Selecciona un armature, cámara o objeto objetivo para evitar ambigüedad.")
            else:
                _notify(self, sc, "ERROR", "No hay un objeto animable válido para generar la animación.")
            return {"CANCELLED"}

        context_objects = _resolve_context_objects(sc, target_obj=target_obj)
        scene_ctx  = scan_scene_context(sc, target_obj=target_obj, context_objects=context_objects)
        scripts_ctx = _build_scripts_context(sc)
        full_prompt = (
            f"Animation/Behavior request: {prompt}\n\n"
            f"{_context_instruction(target_obj, context_objects)}\n\n"
            f"Scene:\n{json.dumps(scene_ctx, separators=(',', ':'))}\n\n"
            + (f"Available scripts:\n{scripts_ctx}\n\n" if scripts_ctx else "")
            + f"Respond ONLY with JSON (animations, behaviors, notes)."
        )

        sc.am3d_last_full_prompt = full_prompt

        _pending_data = {
            "prompt":        prompt,
            "full_prompt":   full_prompt,
            "scene_context": scene_ctx,
            "target_name":   target_obj.name,
            "context_target_names": [obj.name for obj in context_objects],
            "item_name":     f"{target_obj.name}: {prompt[:32]}",
        }

        _active_thread = _RequestThread(
            _build_messages(sc, full_prompt),
            sc.am3d_server_url, sc.am3d_model_name,
            sc.am3d_temperature, sc.am3d_timeout, sc.am3d_retry_limit,
        )
        _active_thread.start()

        _set_status(sc, "INFO", "Enviando a LM Studio...")
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


# ── Cancel Request  ───────────────────────────────────
class AM3D_OT_CancelRequest(Operator):
    bl_idname      = "am3d.cancel_request"
    bl_label       = "Cancelar"
    bl_description = "Cancela la petición HTTP en curso"

    def execute(self, context):
        global _active_thread
        sc = context.scene
        if _active_thread and _active_thread.is_alive():
            _active_thread.cancel()
            _notify(self, sc, "INFO", "Petición cancelada.", status_message="Cancelando petición...")
        else:
            _notify(self, sc, "WARNING", "No hay petición en curso.")
        return {"FINISHED"}


# ── Apply ─────────────────────────────────────────────
class AM3D_OT_ApplyAnimation(Operator):
    bl_idname     = "am3d.apply_animation"
    bl_label      = "Aplicar"
    bl_description = "Aplica la animación seleccionada al armature/objeto activo"

    index: IntProperty(default=-1)

    @staticmethod
    def _requires_primary_target(actions):
        return any(action.get("type", "bone") == "bone" and not str(action.get("object", "")).strip() for action in actions)

    def execute(self, context):
        sc  = context.scene
        idx = self.index if self.index >= 0 else sc.animotion3d_index

        if not (0 <= idx < len(sc.animotion3d_animations)):
            _notify(self, sc, "ERROR", "Selecciona una animación válida.")
            return {"CANCELLED"}

        item = sc.animotion3d_animations[idx]
        try:
            actions = _load_json_list(item.json_payload, "Animations")
            behaviors = _load_json_list(item.behaviors_payload, "Behaviors")
            context_names = _load_string_list(item.context_targets_json, "Context targets")
        except Exception as e:
            _notify(self, sc, "ERROR", f"Payload corrupto: {e}")
            return {"CANCELLED"}

        if not actions and not behaviors:
            _notify(self, sc, "ERROR", "La animación seleccionada no contiene acciones ni behaviors aplicables.")
            return {"CANCELLED"}

        obj = _resolve_target_object(context, preferred_name=item.target_name)
        context_objects = _resolve_context_objects(sc, target_obj=obj, stored_names=context_names)
        if actions and obj is None and self._requires_primary_target(actions):
            target_label = item.target_name or sc.am3d_target_name.strip() or "actual"
            _notify(self, sc, "ERROR", f"No se encontró el objetivo '{target_label}' para aplicar la animación.")
            return {"CANCELLED"}

        bpy.ops.ed.undo_push(message="Animotion3D: Apply Animation")
        warnings = []
        if actions:
            warnings.extend(apply_actions(obj, actions))
        if behaviors:
            warnings.extend(apply_behaviors(sc, obj, behaviors, context_objects=context_objects))
        for w in warnings:
            _notify(self, sc, "WARNING", w, update_status=False)

        applied_actions  = len(actions)  - sum(1 for w in warnings if "frame" in w.lower() and "no encontrado" in w.lower())
        applied_behaviors = len(behaviors) - sum(1 for w in warnings if "behavior" in w.lower())
        item.applied = (applied_actions > 0 or applied_behaviors > 0)

        summary = []
        if actions:
            summary.append(f"{len(actions)} keyframes")
        if behaviors:
            summary.append(f"{len(behaviors)} behaviors")
        target_label = obj.name if obj else "escena"
        warn_str = f" ({len(warnings)} advertencias)" if warnings else ""
        _notify(self, sc, "INFO", f"Aplicado en '{target_label}': {', '.join(summary)}.{warn_str}")
        return {"FINISHED"}


# ── Feedback / Refine ─────────────────────────────────
class AM3D_OT_SendFeedback(Operator):
    bl_idname     = "am3d.send_feedback"
    bl_label      = "Refinar con feedback"
    bl_description = "Detecta cambios manuales y pide a la IA una versión mejorada"

    _timer = None

    def modal(self, context, event):
        global _active_thread
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if not (_active_thread and _active_thread.is_alive()):
            self.cancel(context)
            if _active_thread is None:
                return {"CANCELLED"}

            content = _active_thread.result_content
            error   = _active_thread.result_error
            _active_thread = None

            if error:
                level = "INFO" if "cancelada" in error.lower() else "ERROR"
                _notify(self, context.scene, level, error)
                return {"CANCELLED"}

            return _finish_request(self, context, content)

        context.scene.am3d_status = "⏳ Procesando feedback…"
        return {"PASS_THROUGH"}

    def execute(self, context):
        global _active_thread, _pending_data

        if _active_thread and _active_thread.is_alive():
            _notify(self, context.scene, "WARNING", "Petición en curso, espera.", update_status=False)
            return {"CANCELLED"}

        sc  = context.scene
        idx = sc.animotion3d_index

        if not (0 <= idx < len(sc.animotion3d_animations)):
            _notify(self, sc, "ERROR", "Selecciona una animación para refinar.")
            return {"CANCELLED"}

        item = sc.animotion3d_animations[idx]
        try:
            original_actions = _load_json_list(item.json_payload, "Animations")
            original_behaviors = _load_json_list(item.behaviors_payload, "Behaviors")
            context_names = _load_string_list(item.context_targets_json, "Context targets")
        except Exception as e:
            _notify(self, sc, "WARNING", f"Animations del item dañadas o vacías: {e}", update_status=False)
            original_actions = []
            original_behaviors = []
            context_names = []

        target_obj = _resolve_target_object(context, preferred_name=item.target_name)
        if target_obj is None:
            target_label = item.target_name or sc.am3d_target_name.strip() or "actual"
            _notify(self, sc, "ERROR", f"No se encontró el objetivo '{target_label}' para refinar.")
            return {"CANCELLED"}

        step       = max(1, sc.am3d_feedback_step)
        notes      = sc.am3d_feedback_notes.strip() or "Mejora y suaviza la animación."
        context_objects = _resolve_context_objects(sc, target_obj=target_obj, stored_names=context_names)
        scene_ctx  = scan_scene_context(sc, target_obj=target_obj, context_objects=context_objects)
        reference_ctx = scan_scene_context(sc)

        tracked_property_paths = _collect_feedback_property_paths(
            original_actions,
            target_obj=target_obj,
            context_objects=context_objects,
        )
        current_state = read_keyframes_from_fcurves(
            sc,
            target_obj=target_obj,
            context_objects=context_objects,
        )
        manual_edits  = detect_manual_edits(original_actions, sc, sc.frame_start, sc.frame_end, step, target_obj=target_obj)
        behavior_edits = detect_behavior_edits(original_behaviors, sc, context_objects=context_objects)
        scene_behavior_state = _collect_scene_behavior_state(sc, context_objects=context_objects)

        combined_edits = manual_edits + behavior_edits
        edits_desc = "\n".join(combined_edits) if combined_edits else "No se detectaron cambios significativos."

        full_prompt = (
            f"Refine the animation for target '{target_obj.name}' ({target_obj.type}). Original prompt: '{item.prompt}'\n\n"
            f"User notes: {notes}\n\n"
            f"{_context_instruction(target_obj, context_objects)}\n\n"
            f"Detected manual edits:\n{edits_desc}\n\n"
            f"Current animation keyframes (actual keyframes only, interpolated frames excluded):\n"
            f"{json.dumps(current_state, indent=2)}\n\n"
            f"Scene behavior state:\n{json.dumps(scene_behavior_state, indent=2)}\n\n"
            f"Target details:\n{json.dumps(scene_ctx, indent=2)}\n\n"
            f"Reference scene:\n{json.dumps(reference_ctx, indent=2)}\n\n"
            + (_build_scripts_context(sc) and f"Available scripts:\n{_build_scripts_context(sc)}\n\n" or "")
            + f"Respond with improved JSON (animations, behaviors, notes)."
        )

        sc.am3d_last_full_prompt = full_prompt

        _pending_data = {
            "prompt":        notes,
            "full_prompt":   full_prompt,
            "scene_context": scene_ctx,
            "target_name":   target_obj.name,
            "context_target_names": [obj.name for obj in context_objects],
            "item_name":     f"{target_obj.name}: {notes[:27]}",
        }

        ok, err = save_sft_pair(
            item.prompt,
            scene_ctx,
            {"animations": original_actions, "behaviors": original_behaviors},
            {"sampled_state": current_state, "behavior_state": scene_behavior_state},
        )
        if not ok:
            _notify(self, sc, "WARNING", f"SFT no guardado: {err}", update_status=False)
        else:
            _notify(self, sc, "INFO", f"Par SFT guardado en {SFT_FILENAME}", update_status=False)

        _active_thread = _RequestThread(
            _build_messages(sc, full_prompt),
            sc.am3d_server_url, sc.am3d_model_name,
            sc.am3d_temperature, sc.am3d_timeout, sc.am3d_retry_limit,
        )
        _active_thread.start()

        _set_status(sc, "INFO", "Enviando feedback a LM Studio...")
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


# ── Clear Conversation ────────────────────────────────
class AM3D_OT_ClearConversation(Operator):
    bl_idname     = "am3d.clear_conversation"
    bl_label      = "Limpiar historial"
    bl_description = "Borra el historial de conversación (nuevo contexto para la IA)"

    def execute(self, context):
        context.scene.am3d_conversation.clear()
        _set_status(context.scene, "INFO", "Historial limpiado.")
        return {"FINISHED"}


class AM3D_OT_ClearLogs(Operator):
    bl_idname     = "am3d.clear_logs"
    bl_label      = "Limpiar eventos"
    bl_description = "Borra la bitácora de errores y advertencias"

    def execute(self, context):
        context.scene.am3d_logs.clear()
        context.scene.am3d_log_index = 0
        _set_status(context.scene, "INFO", "Bitácora limpiada.")
        return {"FINISHED"}


# ── Delete Animation ──────────────────────────────────
class AM3D_OT_DeleteAnimation(Operator):
    bl_idname     = "am3d.delete_animation"
    bl_label      = "Eliminar"
    bl_description = "Elimina esta animación de la lista"

    index: IntProperty()

    def execute(self, context):
        sc = context.scene
        if 0 <= self.index < len(sc.animotion3d_animations):
            sc.animotion3d_animations.remove(self.index)
            sc.animotion3d_index = max(0, self.index - 1)
        return {"FINISHED"}


# ── Clear All Animations  ─────────────────────────────
class AM3D_OT_ClearAllAnimations(Operator):
    bl_idname     = "am3d.clear_all_animations"
    bl_label      = "Limpiar lista"
    bl_description = "Elimina TODAS las animaciones generadas de la lista"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        sc = context.scene
        sc.animotion3d_animations.clear()
        sc.animotion3d_index = 0
        _set_status(sc, "INFO", "Lista de animaciones limpiada.")
        return {"FINISHED"}


# ── Preview JSON ──────────────────────────────────────
class AM3D_OT_PreviewJSON(Operator):
    bl_idname     = "am3d.preview_json"
    bl_label      = "Ver JSON"
    bl_description = "Muestra los primeros keyframes en el log de Blender"

    def execute(self, context):
        sc  = context.scene
        idx = sc.animotion3d_index
        if not (0 <= idx < len(sc.animotion3d_animations)):
            _notify(self, sc, "ERROR", "Sin animación seleccionada.")
            return {"CANCELLED"}
        try:
            item = sc.animotion3d_animations[idx]
            actions = _load_json_list(item.json_payload, "Animations")
            behaviors = _load_json_list(item.behaviors_payload, "Behaviors")
            preview = {
                "animations": actions[:5],
                "behaviors": behaviors[:5],
            }
            _notify(self, sc, "INFO", f"Preview disponible. Actions: {len(actions)}, behaviors: {len(behaviors)}", update_status=False)
            self.report({"INFO"}, json.dumps(preview, indent=2))
        except Exception as e:
            _notify(self, sc, "ERROR", str(e))
        return {"FINISHED"}


# ── Load Rig Map ──────────────────────────────────────
class AM3D_OT_LoadRigMap(Operator):
    bl_idname     = "am3d.load_rigmap"
    bl_label      = "Cargar rig map"
    bl_description = "Carga un JSON de mapeo de huesos/objetos (source → target)"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        sc = context.scene
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if not isinstance(mapping, dict):
                _notify(self, sc, "ERROR", "El JSON debe ser un objeto {source: target}.")
                return {"CANCELLED"}
            sc.am3d_rigmap_json = json.dumps(mapping)
            _notify(self, sc, "INFO", f"Rig map cargado: {len(mapping)} entradas desde {os.path.basename(self.filepath)}", update_status=False)
        except Exception as e:
            _notify(self, sc, "ERROR", f"Error cargando rig map: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class AM3D_OT_CaptureContextSelection(Operator):
    bl_idname     = "am3d.capture_context_selection"
    bl_label      = "Usar selección actual"
    bl_description = "Guarda la selección actual como contexto multiobjeto para la IA"

    def execute(self, context):
        sc = context.scene
        selected = _unique_animatable_objects(context.selected_objects)
        if not selected:
            _notify(self, sc, "WARNING", "No hay objetos animables seleccionados.")
            return {"CANCELLED"}

        sc.am3d_context_targets.clear()
        for obj in selected:
            item = sc.am3d_context_targets.add()
            item.name = obj.name
            item.obj_type = obj.type

        if _is_animatable_object(context.object):
            sc.am3d_target_name = context.object.name  # type: ignore[union-attr]

        _notify(self, sc, "INFO", f"Contexto actualizado con {len(selected)} objeto(s).")
        return {"FINISHED"}


class AM3D_OT_ClearContextTargets(Operator):
    bl_idname     = "am3d.clear_context_targets"
    bl_label      = "Limpiar contexto"
    bl_description = "Borra el contexto multiobjeto guardado"

    def execute(self, context):
        context.scene.am3d_context_targets.clear()
        _set_status(context.scene, "INFO", "Contexto multiobjeto limpiado.")
        return {"FINISHED"}


# ── Preview Full Prompt (popup con texto copiable) ────
class AM3D_OT_PreviewPrompt(Operator):
    bl_idname      = "am3d.preview_prompt"
    bl_label       = "Prompt enviado a la IA"
    bl_description = "Muestra el último prompt completo enviado a la IA para revisarlo o copiarlo"

    def invoke(self, context, event):
        if not context.scene.am3d_last_full_prompt:
            self.report({"WARNING"}, "Aún no se ha enviado ningún prompt.")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, width=720)

    def draw(self, context):
        layout = self.layout
        sc = context.scene
        layout.label(text="Último prompt enviado a LM Studio:", icon="TEXT")
        layout.separator(factor=0.5)
        col = layout.column(align=True)
        col.prop(sc, "am3d_last_full_prompt", text="")
        layout.separator(factor=0.5)
        layout.operator("am3d.copy_prompt_clipboard", text="Copiar al portapapeles", icon="COPYDOWN")

    def execute(self, context):
        return {"FINISHED"}


class AM3D_OT_CopyPromptToClipboard(Operator):
    bl_idname      = "am3d.copy_prompt_clipboard"
    bl_label       = "Copiar prompt al portapapeles"
    bl_description = "Copia el último prompt completo al portapapeles del sistema"

    def execute(self, context):
        sc = context.scene
        if not sc.am3d_last_full_prompt:
            self.report({"WARNING"}, "No hay prompt para copiar.")
            return {"CANCELLED"}
        context.window_manager.clipboard = sc.am3d_last_full_prompt
        self.report({"INFO"}, "Prompt copiado al portapapeles.")
        return {"FINISHED"}


# ══════════════════════════════════════════════════════
#  SCRIPTS
# ══════════════════════════════════════════════════════
def _build_scripts_context(sc) -> str:
    """Devuelve los scripts guardados como contexto para el AI, si est\u00e1 activado."""
    if not getattr(sc, "am3d_scripts_use_context", False) or not sc.am3d_scripts:
        return ""
    parts = ["User-defined Blender Python scripts available in this session:"]
    for i, item in enumerate(sc.am3d_scripts):
        parts.append(f"\n--- Script {i + 1}: {item.name} ---")
        if item.description:
            parts.append(f"Purpose: {item.description}")
        parts.append(f"```python\n{item.code}\n```")
    return "\n".join(parts)


class AM3D_OT_RunScript(Operator):
    bl_idname      = "am3d.run_script"
    bl_label       = "Ejecutar script"
    bl_description = "Ejecuta el script actual directamente en Blender"

    def execute(self, context):
        sc   = context.scene
        code = sc.am3d_script_code.strip()
        if not code:
            _notify(self, sc, "ERROR", "El script est\u00e1 vac\u00edo.")
            return {"CANCELLED"}
        try:
            exec(compile(code, "<am3d_script>", "exec"), {"bpy": bpy, "context": context})
            _notify(self, sc, "INFO", "Script ejecutado correctamente.")
        except Exception as e:
            _notify(self, sc, "ERROR", f"Error en script: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}


class AM3D_OT_SaveScript(Operator):
    bl_idname      = "am3d.save_script"
    bl_label       = "Guardar script"
    bl_description = "Guarda el script actual en la biblioteca"

    def execute(self, context):
        sc   = context.scene
        code = sc.am3d_script_code.strip()
        name = sc.am3d_script_name.strip() or "Script sin nombre"
        if not code:
            _notify(self, sc, "ERROR", "El script est\u00e1 vac\u00edo.")
            return {"CANCELLED"}
        item             = sc.am3d_scripts.add()
        item.name        = name
        item.description = sc.am3d_script_desc.strip()
        item.code        = code
        sc.am3d_scripts_index = len(sc.am3d_scripts) - 1
        _notify(self, sc, "INFO", f"Script '{name}' guardado en biblioteca.")
        return {"FINISHED"}


class AM3D_OT_DeleteScript(Operator):
    bl_idname      = "am3d.delete_script"
    bl_label       = "Borrar script"
    bl_description = "Elimina el script seleccionado de la biblioteca"

    def execute(self, context):
        sc  = context.scene
        idx = sc.am3d_scripts_index
        if idx < 0 or idx >= len(sc.am3d_scripts):
            return {"CANCELLED"}
        name = sc.am3d_scripts[idx].name
        sc.am3d_scripts.remove(idx)
        sc.am3d_scripts_index = max(0, idx - 1)
        _notify(self, sc, "INFO", f"Script '{name}' eliminado.")
        return {"FINISHED"}


class AM3D_OT_LoadScriptToEditor(Operator):
    bl_idname      = "am3d.load_script_to_editor"
    bl_label       = "Cargar en editor"
    bl_description = "Carga el script seleccionado en el editor"

    def execute(self, context):
        sc  = context.scene
        idx = sc.am3d_scripts_index
        if idx < 0 or idx >= len(sc.am3d_scripts):
            return {"CANCELLED"}
        item                 = sc.am3d_scripts[idx]
        sc.am3d_script_name  = item.name
        sc.am3d_script_desc  = item.description
        sc.am3d_script_code  = item.code
        return {"FINISHED"}


# ── Open SFT Folder ───────────────────────────────────
class AM3D_OT_OpenSFTFolder(Operator):
    bl_idname     = "am3d.open_sft_folder"
    bl_label      = "Abrir carpeta SFT"
    bl_description = "Abre la carpeta donde se guardan los pares de entrenamiento"

    def execute(self, context):
        import subprocess, sys
        try:
            if sys.platform == "win32":
                os.startfile(DATASET_DIR)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", DATASET_DIR])
            else:
                subprocess.Popen(["xdg-open", DATASET_DIR])
        except Exception as e:
            _notify(self, context.scene, "WARNING", f"No se pudo abrir la carpeta: {e}")
        return {"FINISHED"}

# ══════════════════════════════════════════════════════
#  UI LIST
# ══════════════════════════════════════════════════════
class AM3D_UL_ScriptList(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=item.name[:28], icon="TEXT")
        if item.description:
            row.label(text=item.description[:28])


class AM3D_UL_AnimList(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        ico = "PREFERENCES" if item.anim_type == "behavior" else "CHECKMARK" if item.applied else "ANIM"
        row.label(text=item.name[:38], icon=ico)
        row.operator("am3d.apply_animation",  text="", icon="PLAY").index   = index
        row.operator("am3d.delete_animation", text="", icon="X").index      = index


class AM3D_UL_LogList(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        item_icon = {
            "ERROR": "ERROR",
            "WARNING": "INFO",
            "INFO": "CHECKMARK",
        }.get(item.level, "INFO")
        row.label(text=item.message[:120], icon=item_icon)

# ══════════════════════════════════════════════════════
#  PANEL
# ══════════════════════════════════════════════════════
class AM3D_PT_Panel(Panel):
    bl_label       = "Animotion3D"
    bl_idname      = "AM3D_PT_panel"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "Animotion3D"

    def draw(self, context):
        sc     = context.scene
        layout: UILayout = self.layout  # type: ignore[assignment]

        # ── Estado + botón cancelar ───────────────────
        if sc.am3d_status:
            box = layout.box()
            row = box.row(align=True)
            ico = "ERROR" if sc.am3d_status.startswith("✗") else "INFO"
            row.label(text=sc.am3d_status, icon=ico)
            # Mostrar cancelar solo si hay petición en curso
            if _active_thread and _active_thread.is_alive():
                row.operator("am3d.cancel_request", text="", icon="CANCEL")

        # ── Config LM Studio ──────────────────────────
        col = layout.column(align=True)
        col.label(text="LM Studio", icon="PREFERENCES")
        col.prop(sc, "am3d_server_url",  text="URL")
        col.prop(sc, "am3d_model_name",  text="Modelo")
        row = col.row(align=True)
        row.prop(sc, "am3d_temperature", text="Temp")
        row.prop(sc, "am3d_timeout",     text="Timeout s")
        row = col.row(align=True)
        row.prop(sc, "am3d_retry_limit", text="Reintentos")
        if sc.am3d_retry_limit < 0:
            col.label(text="Reintentos infinitos hasta cancelar", icon="INFO")

        layout.separator()

        # ── Escena detectada (usa caché) ──────────────
        ctx = get_cached_scene_context(sc)
        box = layout.box()
        box.label(text="Escena detectada", icon="SCENE_DATA")
        for arm in ctx.get("armatures", []):
            box.label(text=f"  🦴 {arm['name']} ({len(arm.get('bones', []))} huesos)", icon="ARMATURE_DATA")
        if ctx.get("cameras"):
            box.label(text=f"  📷 {len(ctx['cameras'])} cámara(s)", icon="CAMERA_DATA")
        if ctx.get("lights"):
            box.label(text=f"  💡 {len(ctx['lights'])} luz/luces", icon="LIGHT")
        if ctx.get("objects"):
            box.label(text=f"  📦 {len(ctx['objects'])} objeto(s)", icon="OBJECT_DATA")
        if not any([ctx.get("armatures"), ctx.get("cameras"), ctx.get("lights"), ctx.get("objects")]):
            box.label(text="  (escena vacía)", icon="ERROR")

        layout.separator()

        # ── Rig Mapping ───────────────────────────────
        col = layout.column(align=True)
        col.label(text="Rig Mapping (opcional)", icon="OUTLINER_DATA_ARMATURE")
        # Mostrar resumen del mapping en lugar del JSON crudo
        mapping = _get_rig_mapping(sc)
        if mapping:
            col.label(text=f"  {len(mapping)} entradas cargadas", icon="CHECKMARK")
        else:
            col.label(text="  (sin mapping)", icon="BLANK1")
        col.operator("am3d.load_rigmap", text="Cargar desde archivo…", icon="FILE_FOLDER")

        layout.separator()

        # ── Objetivo explícito ───────────────────────
        box = layout.box()
        box.label(text="Objetivo de animación", icon="RESTRICT_SELECT_OFF")
        box.prop_search(sc, "am3d_target_name", sc, "objects", text="Objeto / rig")
        target_obj = _resolve_target_object(context)
        if target_obj:
            box.label(text=f"Usando: {target_obj.name} ({target_obj.type})", icon=_target_icon(target_obj))  # type: ignore[arg-type]
        elif _count_animatable_objects(sc) > 1:
            box.label(text="Selecciona un objetivo para evitar ambigüedad", icon="ERROR")
        else:
            box.label(text="Si lo dejas vacío, se usa el objeto activo", icon="INFO")

        context_box = layout.box()
        context_box.label(text="Contexto multiobjeto", icon="GROUP")
        row = context_box.row(align=True)
        row.operator("am3d.capture_context_selection", text="Usar selección actual", icon="RESTRICT_SELECT_OFF")
        if sc.am3d_context_targets:
            row.operator("am3d.clear_context_targets", text="", icon="TRASH")
        if sc.am3d_context_targets:
            for item in sc.am3d_context_targets:
                context_box.label(text=f"{item.name} ({item.obj_type})", icon=_target_icon(sc.objects.get(item.name)))  # type: ignore[arg-type]
        else:
            context_box.label(text="Sin contexto extra: se usará solo el objetivo principal.", icon="INFO")
        context_box.label(text="Tip: para escenas complejas, captura varias cámaras u objetos antes de generar.", icon="CAMERA_DATA")

        layout.separator()

        # ── Bitácora de errores/eventos ──────────────
        box = layout.box()
        row = box.row(align=True)
        row.label(text="Errores y eventos", icon="TEXT")
        if sc.am3d_logs:
            row.operator("am3d.clear_logs", text="", icon="TRASH")
        if sc.am3d_logs:
            box.template_list(
                "AM3D_UL_LogList", "",
                sc, "am3d_logs",
                sc, "am3d_log_index",
                rows=5,
            )
        else:
            box.label(text="(sin mensajes aún)", icon="INFO")

        layout.separator()

        # ── Prompt ────────────────────────────────────
        box = layout.box()
        box.label(text="Generar animación", icon="TEXT")
        box.prop(sc, "am3d_prompt_text", text="")

        row = box.row(align=True)
        row.scale_y = 1.5
        if _active_thread and _active_thread.is_alive():
            row.operator("am3d.cancel_request", text="Cancelar petición", icon="CANCEL")
        else:
            row.operator("am3d.generate_animation", icon="SHADERFX", text="Generar con IA")
            row.operator("am3d.preview_prompt", text="", icon="ZOOM_IN")

        conv_n = len(sc.am3d_conversation) // 2
        if conv_n:
            row2 = box.row()
            row2.label(text=f"Historial: {conv_n} turno(s)")
            row2.operator("am3d.clear_conversation", text="Limpiar", icon="TRASH")

        layout.separator()

        # ── Lista animaciones ─────────────────────────
        row = layout.row(align=True)
        row.label(text="Animaciones generadas", icon="NLA")
        if sc.animotion3d_animations:
            row.operator("am3d.clear_all_animations", text="", icon="TRASH")

        layout.template_list(
            "AM3D_UL_AnimList", "",
            sc, "animotion3d_animations",
            sc, "animotion3d_index",
            rows=4,
        )
        if sc.animotion3d_animations:
            row = layout.row(align=True)
            row.operator("am3d.apply_animation", text="Aplicar seleccionada", icon="PLAY").index = -1
            row.operator("am3d.preview_json",    text="",                      icon="ZOOM_ALL")

        layout.separator()

        # ── Feedback ──────────────────────────────────
        box = layout.box()
        box.label(text="Feedback y refinamiento", icon="LOOP_BACK")
        box.label(text="① Edita la animación en el timeline")
        box.label(text="② Describe la mejora deseada")
        box.prop(sc, "am3d_feedback_notes", text="Notas")
        row = box.row(align=True)
        row.prop(sc, "am3d_feedback_step", text="Sample (frames)", slider=True)
        box.operator("am3d.send_feedback", icon="EXPORT", text="Refinar con IA")

        layout.separator()

        # ── SFT Dataset ───────────────────────────────
        box = layout.box()
        box.label(text="Dataset SFT", icon="FILE_SCRIPT")
        sft_exists = os.path.isfile(SFT_FILENAME)
        if sft_exists:
            # FIX v0.6: contar líneas solo una vez por sesión, no en cada redibujado
            box.label(text=f"  {_get_sft_count()} pares guardados")
        else:
            box.label(text="  (sin pares aún)")
        box.operator("am3d.open_sft_folder", text="Abrir carpeta", icon="FOLDER_REDIRECT")

# ══════════════════════════════════════════════════════
#  REGISTER
# ══════════════════════════════════════════════════════
classes = (
    AM3D_ConvMessage,
    AM3D_LogItem,
    AM3D_ContextTargetItem,
    AM3D_AnimItem,
    AM3D_ScriptItem,
    AM3D_OT_GenerateAnimation,
    AM3D_OT_CancelRequest,        # NUEVO
    AM3D_OT_ApplyAnimation,
    AM3D_OT_SendFeedback,
    AM3D_OT_ClearConversation,
    AM3D_OT_ClearLogs,
    AM3D_OT_DeleteAnimation,
    AM3D_OT_ClearAllAnimations,   # NUEVO
    AM3D_OT_PreviewJSON,
    AM3D_OT_LoadRigMap,
    AM3D_OT_CaptureContextSelection,
    AM3D_OT_ClearContextTargets,
    AM3D_OT_PreviewPrompt,
    AM3D_OT_CopyPromptToClipboard,
    AM3D_OT_RunScript,
    AM3D_OT_SaveScript,
    AM3D_OT_DeleteScript,
    AM3D_OT_LoadScriptToEditor,
    AM3D_OT_OpenSFTFolder,
    AM3D_UL_ScriptList,
    AM3D_UL_AnimList,
    AM3D_UL_LogList,
    AM3D_PT_Panel,
)

_SCENE_PROPS = {
    "animotion3d_animations": lambda: CollectionProperty(type=AM3D_AnimItem),
    "am3d_logs":              lambda: CollectionProperty(type=AM3D_LogItem),
    "am3d_conversation":      lambda: CollectionProperty(type=AM3D_ConvMessage),
    "am3d_context_targets":   lambda: CollectionProperty(type=AM3D_ContextTargetItem),
    "am3d_log_index":         lambda: IntProperty(default=0),
    "animotion3d_index":      lambda: IntProperty(default=0),
    "am3d_prompt_text":       lambda: StringProperty(name="Prompt", default=""),
    "am3d_target_name":       lambda: StringProperty(name="Target", default=""),
    "am3d_status":            lambda: StringProperty(default=""),
    "am3d_server_url":        lambda: StringProperty(name="URL", default=DEFAULT_URL),
    "am3d_model_name":        lambda: StringProperty(name="Modelo", default=DEFAULT_MODEL),
    "am3d_temperature":       lambda: FloatProperty(name="Temperatura", default=0.2, min=0.0, max=2.0, step=5),
    "am3d_timeout":           lambda: IntProperty(name="Timeout", default=DEFAULT_TIMEOUT, min=3600, max=10800),
    "am3d_retry_limit":       lambda: IntProperty(name="Reintentos", default=DEFAULT_RETRIES, min=-1, max=9999),
    "am3d_feedback_notes":    lambda: StringProperty(name="Notas", default="Hazla más fluida y natural."),
    "am3d_feedback_step":     lambda: IntProperty(name="Step", default=2, min=1, max=10),
    "am3d_rigmap_json":       lambda: StringProperty(name="Rig Map JSON", default=""),
    "am3d_scripts":           lambda: CollectionProperty(type=AM3D_ScriptItem),
    "am3d_scripts_index":     lambda: IntProperty(default=0),
    "am3d_script_name":       lambda: StringProperty(name="Nombre", default=""),
    "am3d_script_desc":       lambda: StringProperty(name="Descripci\u00f3n", default=""),
    "am3d_script_code":       lambda: StringProperty(name="C\u00f3digo", default=""),
    "am3d_scripts_use_context": lambda: BoolProperty(name="Scripts como contexto IA", default=False),
    "am3d_last_full_prompt":    lambda: StringProperty(name="Último Prompt", default=""),
}


def register():
    for c in classes:
        bpy.utils.register_class(c)
    for name, factory in _SCENE_PROPS.items():
        setattr(bpy.types.Scene, name, factory())


def unregister():
    for name in _SCENE_PROPS:
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
