# Reporte de estado — GalaxyChop

Actualizado: 2026-10-10 · Rama: `dev` · Versión: `1.0.dev0` · Python: `>=3.11,<3.16`

## Estado

- `tox -r`: `py311`–`py314` 268 passed / 1 xfailed; cobertura 96,58 %; `style`, `docstyle`, `check-*` y `make-docs` OK.
- `py315` falla solo porque `h5py` no tiene wheels para 3.15 (se compila y necesita `libhdf5-dev`; el CI ya lo instala).
- El historial de cambios está en `git log` y en `CHANGELOG.md`.

## Pendientes

- Crear el secret `PYPI_API_TOKEN` en GitHub (lo usa `publish.yml`); borrar los viejos `PYPI_USERNAME`/`PYPI_PASSWORD`.
- Verificar el primer run de los workflows nuevos en GitHub Actions.
- Revisar desde lo físico los resultados de `AutoGaussianMixture` (cambiaron con `d3a86c8`).
- Decidir la estrategia de ramas (`dev` vs `master` y ramas viejas) y la versión del release.
- Sacar el paso de `libhdf5-dev` de `tests.yml` cuando `h5py` publique wheels para 3.15.
- Plots: los nombres de componente que no están en `galaxychop.config` se dibujan como "Unclassified"; las leyendas con `loc="best"` generan un aviso de lentitud.
- Cobertura baja: `preproc/potential_energy/__init__.py` (74 %), `preproc/_base.py` (78 %).

## Para retomar

- `git log --oneline -10` y `python -m pytest -q` con el venv (Python 3.13).
- Los tutoriales se ejecutan con datos de `tests/datasets/`; nbsphinx no los re-ejecuta porque tienen salidas guardadas.
