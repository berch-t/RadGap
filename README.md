# RadGap

> Benchmark reproductible de la **généralisation**, de l'**efficacité-label** et de l'**équité**
> des foundation models en radiologie, sur un seul GPU (RTX 4080 Super 16 Go).

> ⚠️ Projet en cours (jalon M0 — bootstrap). Le README complet est généré au jalon M8.
> Voir [`CLAUDE.md`](CLAUDE.md) pour le contexte et [`PLAN.md`](PLAN.md) pour l'avancement.

## Installation (dev)

```bash
uv python install 3.11
uv sync                       # crée .venv avec torch CUDA (cu124)
cp env.example .env           # renseigner RADGAP_DATA_ROOT + AIMI_API_KEY
uv run python scripts/check_env.py   # vérifie GPU + forward AMP
uv run pytest -q
```

## Données

Les données ne sont **jamais** committées (licences recherche Stanford AIMI / PhysioNet).
Voir `.claude/skills/medical-imaging-data` pour l'acquisition et l'harmonisation.

## Licence

Code sous MIT. **Les données et poids de modèles conservent leurs propres licences
recherche non-commerciales** (Stanford Research Use Agreement, PhysioNet credentialing).
