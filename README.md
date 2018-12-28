# mixed_panel_logit_default_risk
I use a mixed panel logit model with randomized coefficients across countries and select year-fixed effects using L2 regularization to predict unconditional probabilities of sovereign external debt default for out-of-sample years. 

For future research, I wish to combine predicted probabilities with estimated "haircut" data to generate a risk score which is interpreted as the unconditional rate of foreign investment loss resulting from default. 

### Dependencies
[pylogit](https://github.com/timothyb0912/pylogit)
