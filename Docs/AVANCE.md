# Bitácora de avance

Se actualiza al cerrar cada sesión de trabajo. El registro va con lo más reciente arriba.
Plan de referencia: [`Plan_de_Accion_VideoTracking.md`](Plan_de_Accion_VideoTracking.md).

## Estado por fase

| Fase | Estado | Nota |
|---|---|---|
| F0 Preparación | En curso | Lo técnico está listo; faltan las tareas del equipo (ver pendientes) |
| F1 Pipeline mínimo | **Hecha** | Video → tracking → conteo por línea → CSV. 6 pruebas pasan |
| F2 Datos propios | Pendiente | Bloqueada por: clips, puntos de captura, guía de anotación |
| F3 Evaluación base | Pendiente | F3a puede arrancar en cuanto se descargue VisDrone |
| F4 Ajuste fino YOLO | Pendiente | |
| F5 Tracking y analítica | Pendiente | |
| F6 Reportes y tablero | Pendiente | |
| F7 Cierre | Pendiente | |

## Pendientes del equipo

- [ ] Crear el usuario administrador de CVAT (comando en el README)
- [ ] Instalar Tailscale en los 3 PCs y configurar `CVAT_HOST` (pasos en el README)
- [ ] Probar que los 3 entran a CVAT desde sus casas (hito de F0)
- [ ] Crear el repo en GitHub y conectar el remoto (`git remote add origin <url>` + `git push -u origin main`)
- [ ] Descargar VisDrone2019-MOT val **desde el navegador** ([link](https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view)) y dejar el zip en `data/visdrone/`
- [ ] Crear la carpeta compartida en Drive (solo material anonimizado)
- [ ] Asignar roles I1, I2 e I3
- [ ] Decidir el umbral mínimo de F1 por clase: 0,60 o 0,65
- [ ] Actualizar la propuesta con el Anexo A del plan
- [ ] PyCharm: volver a seleccionar el intérprete `.venv` (ahora es Python 3.12)

## Registro

### 2026-10-07 — Fase 0 (parte técnica) y Fase 1

**Hecho**
- Entorno: `.venv` recreado con Python 3.12.10; `requirements.txt` fijado (ultralytics 8.4.174, lap 0.5.13, pyyaml 6.0.3).
- Git local inicializado con `.gitignore`: datos, videos, resultados y pesos nunca van a git.
- CVAT v2.78.0 en `C:\Users\USER\cvat`, corriendo en http://localhost:8080. Consume unos 2,9 GB de RAM.
- Video de ejemplo descargado en `data/samples/vehicles.mp4`: autopista en 4K, 25 fps, 538 cuadros.
- Pipeline F1 con arquitectura limpia (`vtia/domain`, `application`, `adapters`, `cli.py`) y pruebas de la lógica de conteo (`tests/test_counting.py`).
- Plan actualizado: arquitectura (§1.4) y decisiones (§0).

**Decisiones**
- Arquitectura limpia pragmática; nombres en inglés en el código.
- Modelo principal: YOLO26n.
- Un track cuenta como máximo una vez por línea, para que el temblor sobre la línea no duplique conteos.
- La clase de un cruce es la clase mayoritaria del track, no la del cuadro del cruce.

**Resultados (video de ejemplo, CPU Ryzen 5 9600X)**

| Modelo | Cruces | Velocidad | Observación |
|---|---|---|---|
| YOLO26n | 8 | 29,1 fps | Pierde 1: duplica el carro negro y el tracker le pasa ese ID al carro blanco, que ya no se cuenta |
| YOLO11n | 10 | 28,1 fps | Cuenta 2 veces el carro negro (dos tracks cruzan en el mismo cuadro) |

Conteo real estimado: 9, por revisión visual de los cuadros de cada evento; el video no se revisó completo. El camión del inicio no cuenta porque arranca ya pasada la línea.

**Hallazgos**
- **TrackEval** (último commit de 2022) usa `np.float`, `np.int` y `np.bool`, que numpy 2 eliminó (81 usos). Hay que decidirlo en F3a: parche, fork mantenido o `py-motmetrics`.
- **VisDrone desde Google Drive:** `gdown` falla porque el archivo superó la cuota de descargas. Hay que bajarlo desde el navegador.
- Las detecciones duplicadas de un mismo vehículo son la principal fuente de error de conteo observada. Se ataca en F3 y F5, con datos de referencia: umbral de confianza, BoT-SORT, supresión de duplicados.

**Siguiente**
- Equipo: cerrar los pendientes de F0.
- F3a: módulo de evaluación con VisDrone, cuando esté descargado.
- F2: protocolo de captura y guía de anotación, cuando definan ciudad, puntos y celulares.
