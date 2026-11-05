from vulcantech.decision import decision_record


def test_pending_without_approved_rule():
    result = decision_record({"class_id": 2}, {"probability": 0.99})
    assert result["action"] is None
    assert result["validation_state"] == "PENDENTE_APROVACAO"
    assert result["equipment_controlled"] is False


def test_invalid_data_stays_pending():
    result = decision_record({"status": "NAO_CLASSIFICADA"}, valid=False)
    assert result["action"] is None
    assert result["validation_state"] == "PENDENTE_DADOS_INVALIDOS"


def test_ambiguous_diagnosis_never_selects_action():
    result = decision_record({"status": "AMBIGUO"}, {"probabilities": {"a": 0.5, "b": 0.5}})
    assert result["action"] is None


def test_conflicting_indicators_do_not_create_unapproved_rule():
    result = decision_record({"class_id": 2}, {"rule_a": "INTERROMPER_EMERGENCIA",
                                               "rule_b": "MANTER_OPERACAO"})
    assert result["action"] is None
    assert result["validation_state"] == "PENDENTE_APROVACAO"
