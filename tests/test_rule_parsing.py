import pytest
from simpful import rule_parsing

def test_postparse():
    """Check that the output is equal to our expected output"""
    normal_rule = "IF (OXI IS low_flow) THEN (POWER IS LOW_POWER)"
    expected_out_normal = ('POWER', 'LOW_POWER')
    output = rule_parsing.postparse(normal_rule)
    assert output == expected_out_normal

    proba_rule = "IF (OXI IS low_flow) THEN P(POWER IS LOW_POWER)=0.33, P(POWER IS MEDIUM_POWER)=0.33, P(POWER IS HIGH_FUN)=0.34"
    expected_out_proba = ([0.33, 0.33, 0.34], ['POWER', 'LOW_POWER', '0.33', 'POWER', 'MEDIUM_POWER', '0.33', 'POWER', 'HIGH_FUN', '0.34'])
    output_proba = rule_parsing.postparse(proba_rule)
    assert output_proba == expected_out_proba

def test_preparse():
    """Check that the output is equal to our expected output"""
    unparsed = "IF (OXI IS low_flow) THEN (POWER IS LOW_POWER)"
    expected_preparsed = '(OXI IS low_flow)'
    output_preparsed = rule_parsing.preparse(unparsed)
    assert output_preparsed == expected_preparsed