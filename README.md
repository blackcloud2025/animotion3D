# Animotion3D AI Director

Proyecto local para desarrollar el add-on de Blender fuera de Downloads.

## Estructura

- `animotion3d_ai_director/__init__.py`: entrypoint del add-on para Blender.
- `.venv-blender/`: entorno local solo para analisis de VS Code.
- `.vscode/settings.json`: configuracion de Pylance para Blender.
- `pyrightconfig.json`: overrides para reducir falsos positivos.
- `scripts/build_zip.ps1`: empaqueta el add-on en `dist/animotion3d_ai_director.zip`.

## Flujo recomendado

1. Abre esta carpeta o el workspace `animotion3d_ai_director_project.code-workspace` en VS Code.
2. Edita `animotion3d_ai_director/__init__.py`.
3. Ejecuta `scripts/build_zip.ps1` para generar el zip instalable en Blender.
4. Instala el zip desde Blender: Edit > Preferences > Add-ons > Install.

## Nota sobre el entorno

El entorno `.venv-blender` no es para ejecutar Blender. Solo contiene stubs (`fake-bpy-module`) para que VS Code entienda `bpy` y reduzca falsos positivos.
