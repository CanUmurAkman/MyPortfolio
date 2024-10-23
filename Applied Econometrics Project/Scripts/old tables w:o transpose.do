*OLD NON-TRANSPOSED TABLES

import excel "/Users/macbookpro/Desktop/ECON426 Research Project/Gasoline.xlsx", sheet("Gasoline.txt") firstrow


*SET PANEL VARIABLE
encode COUNTRY, gen(numerical_country)
xtset numerical_country YEAR

*-----------------------------------------------------------------------------
*PART I: MODELS I&II

*MODEL I 
reg LGASPCAR LINCOMEP LRPMG LCARPCAP
eststo MODELI

*MODEL II (without the lagged dependent variable) = MODEL I
reg LGASPCAR LINCOMEP LRPMG LCARPCAP

*Let's also include the lagged dependent variable for MODEL II
reg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR

*-----------------------------------------------------------------------------
*PART II:Estimation Techniques for MODEL I & TABLE II
*Estimation Techniques for MODEL I
*OLS
reg LGASPCAR LINCOMEP LRPMG LCARPCAP
eststo OLS_M1
*Within Regression (Fixed Effect)
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP, fe
eststo Within_M1
*Between Regression
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP, be
eststo Between_M1
*Random Effect 1: Amemiya (2SGLS) 
xthtaylor LGASPCAR LINCOMEP LRPMG LCARPCAP numerical_country, endog(LRPMG LCARPCAP) am
eststo RE_Amemiya_M1
*Random Effect 2: Swamy&Arora (2SGLS)
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP, re theta
eststo RE_SwamyArora_M1
*TABLE II
estout Within_M1 Between_M1 RE_Amemiya_M1 RE_SwamyArora_M1, cells(b(star fmt(%9.4f)) se(par) t(par) p(par))stats(N r2_a r2_b r2_w r2_o, fmt(%9.4f %9.0g)labels (N))

esttab OLS_M1 Within_M1 Between_M1 RE_Amemiya_M1 RE_SwamyArora_M1 using BASITtabloiki.log, nonumbers mtitles(OLS Within Between Amemiya SwamyArora) cells(b(star fmt(%9.4f)) se(par) t(par) p(par))stats(N r2_a r2_b r2_w r2_o, fmt(%9.4f %9.0g)labels (N)) 

*-----------------------------------------------------------------------------
*PART III: Estimation Techniques for MODEL II & TABLE III
*Estimation Techniques for MODEL II
*OLS
reg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR
eststo OLS_M2
*Within Regression (Fixed Effect)
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR, fe
eststo Within_M2
*Random Effect Model 1: Amemiya (2SGLS) 
xthtaylor LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR numerical_country, endog(LRPMG LCARPCAP) am
eststo RE_Amemiya_M2
*Random Effect Model 2: Swamy&Arora (2SGLS)
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR, re theta
eststo RE_SwamyArora_M2
*TABLE III
estout OLS_M2 Within_M2 RE_Amemiya_M2 RE_SwamyArora_M2, cells(b(star fmt(%9.4f)) se(par) t(par) p(par))stats(N r2_a r2_b r2_w r2_o, fmt(%9.4f %9.0g)labels (N))

esttab OLS_M2 Within_M2 RE_Amemiya_M2 RE_SwamyArora_M2 using BASITtabloüç.log, nonumbers mtitles(OLS Within Amemiya SwamyArora) cells(b(star fmt(%9.4f)) se(par) t(par) p(par))stats(N r2_a r2_b r2_w r2_o, fmt(%9.4f %9.0g)labels (N))



