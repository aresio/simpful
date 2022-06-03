from simpful import rule_parsing

def test_postparse():
    """Check that the output is equal to the expected output"""
    normal_rule = "IF (OXI IS low_flow) THEN (POWER IS LOW_POWER)"
    expected_out_normal = ('POWER', 'LOW_POWER')
    output = rule_parsing.postparse(normal_rule)
    assert output == expected_out_normal

    proba_rule = "IF (OXI IS low_flow) THEN P(POWER is LOW_POWER)=0.33, P(POWER is MEDIUM_POWER)=0.33, P(POWER is HIGH_FUN)=0.34"
    expected_out_proba = ('POWER', 'LOW_POWER', '0.33', 'POWER', 'MEDIUM_POWER', '0.33', 'POWER', 'HIGH_FUN', '0.34')
    output = rule_parsing.postparse(proba_rule)
    print(output)
    print(expected_out_proba)
    assert output == expected_out_proba
