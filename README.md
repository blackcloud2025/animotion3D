# Animotion3D AI Director

Add-on de Blender que genera animaciones mediante IA local usando LM Studio.
El proyecto se edita en VS Code y se instala en Blender como archivo `.zip`.

---

## Estructura del proyecto

| Archivo / Carpeta | Descripción |
|---|---|
| `animotion3d_ai_director/__init__.py` | Código principal del add-on |
| `.venv-blender/` | Entorno virtual solo para VS Code (no ejecuta Blender) |
| `.vscode/settings.json` | Configuracion de Pylance para reconocer `bpy` |
| `pyrightconfig.json` | Suprime falsos positivos del analizador de tipos |
| `scripts/build_zip.ps1` | Empaqueta el add-on en `dist/animotion3d_ai_director.zip` |

---

## Por que existe `.venv-blender` (el modulo falso)

`bpy` es el modulo interno de Blender — no existe en Python normal.
Cuando VS Code analiza tu codigo y ve `import bpy`, marcaria un error falso.

Para evitarlo, `.venv-blender` contiene `fake-bpy-module`: una coleccion de
**stubs** (definiciones de tipos) que imitan la API de `bpy`, `mathutils`, etc.
VS Code los usa solo para analisis; el add-on real siempre corre dentro de Blender.

```
VS Code analiza __init__.py
  └─ ve "import bpy"
  └─ busca en .venv-blender/ → fake-bpy-module → stubs ✅
  └─ sin errores, autocompletado funciona ✅

Blender carga __init__.py
  └─ bpy es el modulo real → todo funciona ✅
```

---

## Como generar e instalar el add-on en Blender

### 1. Generar el ZIP

En el terminal de VS Code:

```powershell
.\scripts\build_zip.ps1
```

Esto crea `dist/animotion3d_ai_director.zip`.

### 2. Instalar en Blender

1. Abre Blender
2. **Edit → Preferences → Add-ons → Install...**
3. Selecciona `dist/animotion3d_ai_director.zip`
4. Activa el checkbox del add-on **"Animotion3D - AI Director"**
5. El panel aparece en **View3D → Sidebar (tecla N) → pestaña "Animotion3D"**

---

## Como conectar con LM Studio

El add-on se comunica con LM Studio a traves de su servidor local.
La URL por defecto ya esta configurada en el codigo:

```
http://127.0.0.1:1234/v1/chat/completions
```

### Pasos en LM Studio

1. Abre LM Studio y carga un modelo (Llama, Mistral, etc.)
2. Ve a la seccion **"Local Server"** (icono de servidor en la barra izquierda)
3. Haz clic en **"Start Server"** — debe mostrar `Server running on port 1234`

### Pasos en Blender

- El campo **URL** ya viene precargado con la direccion correcta
- El campo **Model** puede dejarse como `local-model` o escribir el nombre exacto
  del modelo activo en LM Studio (LM Studio usa el modelo cargado de todas formas)

### Flujo completo

```
LM Studio corriendo con modelo cargado
         ↓
Blender con add-on instalado y activo
         ↓
View3D → N → Animotion3D → escribes prompt → Generate
         ↓
Add-on hace POST a http://127.0.0.1:1234/v1/chat/completions
         ↓
LM Studio responde JSON → add-on aplica keyframes en la escena
```

> **Importante:** LM Studio debe estar corriendo *antes* de presionar Generate.
> Ambos programas deben estar en la misma maquina (localhost).
