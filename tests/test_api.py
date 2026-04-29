from fastapi.testclient import TestClient
from src.api import app

def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] in ["ok", "not_ready"]

def test_triage_returns_expected_shape():
    payload = {
        "claim": {
            "claim_id": "CLM_TEST_UT_001",
            "state": "GA",
            "plan_type": "PPO",
            "provider_specialty": "Ortho",
            "place_of_service": "Office",
            "claim_amount": 2500,
            "out_of_network": False,
            "inpatient_flag": False,
            "prior_auth_required": False,
            "review_risk_score": 0.2,
            "clinical_notes": "member requests prior authorization for medication",
            "call_center_notes": "follow up on oncology rx"
        }
    }

    with TestClient(app) as client:
        r = client.post("/triage", json=payload)
        assert r.status_code == 200

        js = r.json()
        assert "claim_id" in js
        assert "route" in js
        assert "risk" in js and "fraud_score" in js["risk"] and "auto_score" in js["risk"]
        assert "guardrails" in js
        assert "evidence_pack" in js and len(js["evidence_pack"]) > 0
        assert "audit" in js and "routing_policy" in js["audit"]