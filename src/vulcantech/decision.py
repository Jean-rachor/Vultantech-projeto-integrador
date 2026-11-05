from datetime import datetime, timezone
from uuid import uuid4

RULE_VERSION = "pendente-de-aprovacao"


def decision_record(diagnosis: dict, indicators: dict | None = None, *, valid: bool = True) -> dict:
    """Registra o diagnóstico sem convertê-lo em comando operacional não aprovado."""
    return {
        "inspection_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnosis": diagnosis,
        "indicators": indicators or {},
        "criterion_version": RULE_VERSION,
        "criterion_sources": ["Docs/Projeto/Projeto Integrador Avançado certoo.pdf",
                              "Docs/Projeto/Prática tabela.xlsx",
                              "src/vulcantech/decision.py"],
        "validation_state": "PENDENTE_APROVACAO" if valid else "PENDENTE_DADOS_INVALIDOS",
        "action": None,
        "rationale": "Não há regra operacional aprovada nos documentos ou no código consultados.",
        "equipment_controlled": False,
    }
