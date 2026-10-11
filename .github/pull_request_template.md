> **Escribir para quien revisa.** La descripción debe leerse en un minuto: qué cambia, por
> qué, y qué mirar con atención. Sin tablas, sin secciones decorativas, sin repetir lo que
> el diff ya muestra. Si no entra en pocas líneas, el PR es demasiado grande: dividirlo.
> El detalle técnico va en el issue o en el mensaje del commit.
>
> **Un diagrama cuando explique mejor que el texto.** Para flujos, orden de ejecución o
> relaciones entre componentes, un diagrama reemplaza párrafos: incluirlo en lugar de
> describirlos en prosa, no además de.

## 📝 Descripción
<!-- Resumen de los cambios realizados siguiendo el formato del commit template -->
<!-- Ejemplo: :bug: fix(analyzer): mostrar el error que Claude escribe en stdout -->

**Contexto detallado:**
<!-- Qué cambia y por qué, en pocas líneas. Mencionar incidencias vinculadas (Closes #123) -->

---

## 🧪 Verificación Técnica
- [ ] **Tests locales**: He ejecutado `make test` y han pasado con éxito.
- [ ] **Lint y formato**: `make lint` pasa sin errores y el código nuevo respeta `ruff format`.
- [ ] **Dependencias**: ¿Se ha modificado `pyproject.toml`? (Si es sí, reinstalar con `uv pip install --python venv/bin/python -e ".[gpu,dev]"`, con `torch` desde el índice `cu128`).
- [ ] **Validación con audio real**: Si el cambio toca transcriptores, diarizador o los clientes de Codex/Claude (no tienen tests unitarios), ¿se corrió `make process FILE=...` sobre un archivo real?
- [ ] **Higiene**: No se suben `.env`, `output/`, `temp_*.wav`, `temp_chunks/`, `venv/` ni nada de `Audios/`.

---

## 📸 Anexos (Opcional)
<!-- Adjuntar el log del pipeline (`make start` / `make process`) o la nota de Obsidian generada si aplica -->
