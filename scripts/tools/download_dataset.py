"""Descarga el paquete de HF usando una cuenta con acceso aprobado."""
import argparse
import sys
from pathlib import Path

REPO_ID = 'lucasikruger/vis-iris-pupil-segmentation-data'
DATASET_URL = f'https://huggingface.co/datasets/{REPO_ID}'
ACCESS_HELP = f'''Este dataset tiene acceso a pedido con aprobación manual.
1. Pedí acceso en {DATASET_URL}
   Completá nombre, institución y uso previsto; aceptá el uso no comercial
   y no identificar a las personas. Esperá la aprobación manual.
2. Ejecutá: hf auth login
   Usá la misma cuenta aprobada y un token con permiso de lectura del dataset.
3. Volvé a ejecutar este script. Iniciar sesión no reemplaza la aprobación.
Si tu solicitud está pendiente o fue rechazada, no podrás descargar los archivos.'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=ACCESS_HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path('dataset-package'))
    parser.add_argument('--revision', default='main', help='Rama, tag o commit del dataset')
    args = parser.parse_args(argv)
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError
    except ImportError:
        print('Instalá las dependencias: python -m pip install -r requirements.txt', file=sys.stderr)
        return 1
    try:
        # Explicit auth preflight also prevents a cached directory from hiding a denial.
        HfApi().auth_check(REPO_ID, repo_type='dataset', token=True)
        path = snapshot_download(REPO_ID, repo_type='dataset', token=True,
                                 local_dir=args.output, revision=args.revision)
    except (LocalTokenNotFoundError, GatedRepoError):
        print(ACCESS_HELP, file=sys.stderr)
        return 2
    except HfHubHTTPError as error:
        status = getattr(error.response, 'status_code', None)
        if status in (401, 403):
            print(ACCESS_HELP, file=sys.stderr)
            return 2
        print(f'No se pudo descargar el dataset (HTTP {status}). Revisá la conexión, '
              'la revisión solicitada y el estado de Hugging Face antes de reintentar.', file=sys.stderr)
        return 1
    except Exception as error:
        # Do not echo request headers, tokens or server response bodies.
        print(f'No se pudo completar la descarga ({type(error).__name__}). '
              'Revisá la conexión y volvé a intentarlo.', file=sys.stderr)
        return 1
    print(f'Dataset descargado en {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
