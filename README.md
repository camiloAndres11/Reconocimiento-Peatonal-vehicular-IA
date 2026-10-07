# VTIA — Video tracking e IA urbana

Prototipo que detecta, sigue y cuenta actores viales (peatón, bicicleta, motocicleta, automóvil, bus, camión) en video de cámara fija y genera reportes.

- Plan: [`Docs/Plan_de_Accion_VideoTracking.md`](Docs/Plan_de_Accion_VideoTracking.md)
- Avance: [`Docs/AVANCE.md`](Docs/AVANCE.md)

## Instalación (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Video de ejemplo (35 MB):

```powershell
mkdir data\samples
curl.exe -Lo data\samples\vehicles.mp4 https://media.roboflow.com/supervision/video-examples/vehicles.mp4
```

## Uso

```powershell
python -m vtia.cli data\samples\vehicles.mp4 --lines configs\points\sample_vehicles.yaml
```

Escribe `outputs\<video>\events.csv` (un cruce por fila) y `counts.csv` (totales por línea, sentido y clase). Opciones: `--weights yolo11n.pt`, `--tracker botsort.yaml`, `--conf 0.3`, `--out carpeta`.

Cada punto de cámara necesita su YAML en `configs/points/` con las líneas virtuales; ver `sample_vehicles.yaml`.

## Pruebas

```powershell
python -m unittest discover -s tests
```

## Estructura

```
vtia/
├── domain/        reglas puras, sin librerías externas
│   ├── classes.py      6 clases + mapeo COCO
│   ├── models.py       Detection, Line, CrossingEvent
│   └── counting.py     cruce de línea, sentido, clase por mayoría
├── application/
│   └── process_video.py  caso de uso + puerto Tracker
├── adapters/      todo lo que toca librerías o archivos
│   ├── ultralytics_tracker.py  YOLO/RT-DETR + ByteTrack/BoT-SORT
│   ├── yaml_config.py          configs/points/*.yaml -> Line
│   └── csv_writer.py
└── cli.py         punto de entrada; conecta adaptadores y caso de uso
```

Regla de dependencias: `domain` no importa nada del proyecto fuera de sí mismo; `application` solo importa `domain`; `adapters` y `cli` pueden importar todo.

## CVAT (anotación)

Corre en el PC servidor, en `C:\Users\USER\cvat` (v2.78.0).

| Acción | Comando (en `C:\Users\USER\cvat`) |
|---|---|
| Encender | `docker compose up -d` |
| Apagar | `docker compose stop` |
| Crear administrador (una vez) | `docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'` |

Interfaz: http://localhost:8080

**Acceso del equipo por Tailscale:**
1. Los 3 instalan [Tailscale](https://tailscale.com/download) y entran a la misma red (el dueño del servidor invita a los otros dos).
2. En el PC servidor, crear `C:\Users\USER\cvat\.env` con la línea `CVAT_HOST=<IP de Tailscale del servidor>` y ejecutar `docker compose up -d`. Desde ese momento CVAT se abre en `http://<IP de Tailscale>:8080`, también en el propio servidor.
3. Si no carga desde otro PC, permitir el puerto 8080 en el Firewall de Windows del servidor.

**Etiquetas del proyecto** (en CVAT: Projects → + → Raw, pegar):

```json
[
  {"name": "pedestrian", "type": "rectangle", "attributes": []},
  {"name": "bicycle", "type": "rectangle", "attributes": []},
  {"name": "motorcycle", "type": "rectangle", "attributes": []},
  {"name": "car", "type": "rectangle", "attributes": []},
  {"name": "bus", "type": "rectangle", "attributes": []},
  {"name": "truck", "type": "rectangle", "attributes": []}
]
```
