import excel "/Users/macbookpro/Desktop/ECON426 Research Project/Gasoline.xlsx", sheet("Gasoline.txt") firstrow


*SET PANEL VARIABLE
encode COUNTRY, gen(numerical_country)
xtset numerical_country YEAR

*Install asdoc to create a descriptive statistics table
*ssc install asdoc

*DESCRIPTIVE STATISTICS
asdoc sum  LGASPCAR LINCOMEP LRPMG LCARPCAP, replace, save(desc_stat.doc) 

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
scalar b1a = _b[LINCOMEP]
scalar b2a = _b[LRPMG]
scalar b3a = _b[LCARPCAP]
scalar constanta = _b[_cons]
*Within Regression (Fixed Effect)
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP, fe
eststo Within_M1
scalar b1b = _b[LINCOMEP]
scalar b2b = _b[LRPMG]
scalar b3b = _b[LCARPCAP]
scalar constantb = _b[_cons]
*Between Regression
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP, be
eststo Between_M1
scalar b1c = _b[LINCOMEP]
scalar b2c = _b[LRPMG]
scalar b3c = _b[LCARPCAP]
scalar constantc = _b[_cons]
*Random Effect Model 1: Amemiya (2SGLS) 
xthtaylor LGASPCAR LINCOMEP LRPMG LCARPCAP numerical_country, endog(LRPMG LCARPCAP) am
eststo RE_Amemiya_M1
scalar b1d = _b[LINCOMEP]
scalar b2d = _b[LRPMG]
scalar b3d = _b[LCARPCAP]
scalar constantd = _b[_cons]
*Random Effect Model 2: Swamy&Arora (2SGLS)
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP, re theta
eststo RE_SwamyArora_M1
scalar b1e = _b[LINCOMEP]
scalar b2e = _b[LRPMG]
scalar b3e = _b[LCARPCAP]
scalar constante = _b[_cons]
*TABLE II
* Create a Row Vector for the Coefficients
matrix coefficients = (b1a, b2a, b3a, constanta, b1b, b2b, b3b, constantb, b1c, b2c, b3c, constantc, b1d, b2d, b3d, constantd, b1e, b2e, b3e, constante)

* Create a Table of the Coefficients
matrix tabloiki = J(5, 4, .)
matrix tabloiki[1, 1] = coefficients[1, 1]
matrix tabloiki[1, 2] = coefficients[1, 2]
matrix tabloiki[1, 3] = coefficients[1, 3]
matrix tabloiki[1, 4] = coefficients[1, 4]
matrix tabloiki[2, 1] = coefficients[1, 5]
matrix tabloiki[2, 2] = coefficients[1, 6]
matrix tabloiki[2, 3] = coefficients[1, 7]
matrix tabloiki[2, 4] = coefficients[1, 8]
matrix tabloiki[3, 1] = coefficients[1, 9]
matrix tabloiki[3, 2] = coefficients[1, 10]
matrix tabloiki[3, 3] = coefficients[1, 11]
matrix tabloiki[3, 4] = coefficients[1, 12]
matrix tabloiki[4, 1] = coefficients[1, 13]
matrix tabloiki[4, 2] = coefficients[1, 14]
matrix tabloiki[4, 3] = coefficients[1, 15]
matrix tabloiki[4, 4] = coefficients[1, 16]
matrix tabloiki[5, 1] = coefficients[1, 17]
matrix tabloiki[5, 2] = coefficients[1, 18]
matrix tabloiki[5, 3] = coefficients[1, 19]
matrix tabloiki[5, 4] = coefficients[1, 20]

* Add Row and Column Labels
matrix rownames tabloiki = "OLS" "Within" "Between" "Amemiya" "Swamy&Arora"
matrix colnames tabloiki = "LINCOMEP" "LRPMG" "LCARPCAP" "Const."

* Display the Matrix Table
matrix list tabloiki

*Save the Table
log using "tabloiki.log", replace
matrix list tabloiki
log close

*-----------------------------------------------------------------------------
*PART III: Estimation Techniques for MODEL II & TABLE III
*Estimation Techniques for MODEL II
*OLS
reg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR
eststo OLS_M2
scalar b1aa = _b[LINCOMEP]
scalar b2aa = _b[LRPMG]
scalar b3aa = _b[LCARPCAP]
scalar lambdaaa = _b[L.LGASPCAR]
scalar constantaa = _b[_cons]
*Within Regression (Fixed Effect)
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR, fe
eststo Within_M2
scalar b1bb = _b[LINCOMEP]
scalar b2bb = _b[LRPMG]
scalar b3bb = _b[LCARPCAP]
scalar lambdabb = _b[L.LGASPCAR]
scalar constantbb = _b[_cons]
*Random Effect Model 1: Amemiya (2SGLS) 
xthtaylor LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR numerical_country, endog(LRPMG LCARPCAP) am
eststo RE_Amemiya_M2
scalar b1cc = _b[LINCOMEP]
scalar b2cc = _b[LRPMG]
scalar b3cc = _b[LCARPCAP]
scalar lambdacc = _b[L.LGASPCAR]
scalar constantcc = _b[_cons]
*Random Effect Model 2: Swamy&Arora (2SGLS)
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR, re theta
eststo RE_SwamyArora_M2
scalar b1dd = _b[LINCOMEP]
scalar b2dd = _b[LRPMG]
scalar b3dd = _b[LCARPCAP]
scalar lambdadd = _b[L.LGASPCAR]
scalar constantdd = _b[_cons]
*TABLE III
* Create a Matrix for the Coefficients
matrix coefficients = (b1aa, b2aa, b3aa, lambdaaa, constantaa, b1bb, b2bb, b3bb, lambdabb, constantbb, b1cc, b2cc, b3cc, lambdacc, constantcc, b1dd, b2dd, b3dd, lambdadd, constantdd)

* Create a Table of the Coefficients
matrix tabloüç = J(4, 5, .)
matrix tabloüç[1, 1] = coefficients[1, 1]
matrix tabloüç[1, 2] = coefficients[1, 2]
matrix tabloüç[1, 3] = coefficients[1, 3]
matrix tabloüç[1, 4] = coefficients[1, 4]
matrix tabloüç[1, 5] = coefficients[1, 5]
matrix tabloüç[2, 1] = coefficients[1, 6]
matrix tabloüç[2, 2] = coefficients[1, 7]
matrix tabloüç[2, 3] = coefficients[1, 8]
matrix tabloüç[2, 4] = coefficients[1, 9]
matrix tabloüç[2, 5] = coefficients[1, 10]
matrix tabloüç[3, 1] = coefficients[1, 11]
matrix tabloüç[3, 2] = coefficients[1, 12]
matrix tabloüç[3, 3] = coefficients[1, 13]
matrix tabloüç[3, 4] = coefficients[1, 14]
matrix tabloüç[3, 5] = coefficients[1, 15]
matrix tabloüç[4, 1] = coefficients[1, 16]
matrix tabloüç[4, 2] = coefficients[1, 17]
matrix tabloüç[4, 3] = coefficients[1, 18]
matrix tabloüç[4, 4] = coefficients[1, 19]
matrix tabloüç[4, 5] = coefficients[1, 20]

* Add Row and Column Labels
matrix rownames tabloüç = "OLS" "Within" "Amemiya" "Swamy&Arora"
matrix colnames tabloüç = "LINCOMEP" "LRPMG" "LCARPCAP" "L.LGASPCAR" "Const."

* Display the Matrix Table
matrix list tabloüç

*Save the Table
log using "tabloüç.log", replace
matrix list tabloüç
log close

*-----------------------------------------------------------------------------
*PART IV: Long Run Elasticity Estimators for MODEL II & TABLE IV
*OLS Long Run Elasticity Estimators: MODEL II 
reg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR
scalar b1aae = _b[LINCOMEP]
scalar b2aae = _b[LRPMG]
scalar b3aae = _b[LCARPCAP]
scalar lambdaaae = _b[L.LGASPCAR]
scalar OLS_per_capita_income_elasticity = b1aae*(1/(1-lambdaaae))
scalar OLS_gas_price_elasticity = b2aae*(1/(1-lambdaaae))
scalar OLS_cars_per_capita_elasticity = b3aae*(1/(1-lambdaaae))
*Within Regression (Fixed Effect) Long Run Elasticity Estimators: MODEL II
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR, fe
scalar b1bbe = _b[LINCOMEP]
scalar b2bbe = _b[LRPMG]
scalar b3bbe = _b[LCARPCAP]
scalar lambdabbe = _b[L.LGASPCAR]
scalar FE_per_capita_income_elasticity = b1bbe*(1/(1-lambdabbe))
scalar FE_gas_price_elasticity = b2bbe*(1/(1-lambdabbe))
scalar FE_cars_per_capita_elasticity = b3bbe*(1/(1-lambdabbe))
*Random Effect Model 1: Amemiya (2SGLS) Long Run Elasticity Estimators: MODEL II 
xthtaylor LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR numerical_country, endog(LRPMG LCARPCAP) am
scalar b1cce = _b[LINCOMEP]
scalar b2cce = _b[LRPMG]
scalar b3cce = _b[LCARPCAP]
scalar lambdacce = _b[L.LGASPCAR]
scalar Ame_per_capita_income_elasticity = b1cce*(1/(1-lambdacce))
scalar Ame_gas_price_elasticity = b2cce*(1/(1-lambdacce))
scalar Ame_cars_per_capita_elasticity = b3cce*(1/(1-lambdacce))
*Random Effect Model 2: Swamy&Arora (2SGLS) Long Run Elasticity Estimators: MODEL II
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR, re theta
scalar b1dde = _b[LINCOMEP]
scalar b2dde = _b[LRPMG]
scalar b3dde = _b[LCARPCAP]
scalar lambdadde = _b[L.LGASPCAR]
scalar SA_per_capita_income_elasticity = b1dde*(1/(1-lambdadde))
scalar SA_gas_price_elasticity = b2dde*(1/(1-lambdadde))
scalar SA_cars_per_capita_elasticity = b3dde*(1/(1-lambdadde))
*TABLE IV
* Create a Matrix of the Elasticities
matrix elasticity = (OLS_per_capita_income_elasticity, OLS_gas_price_elasticity, OLS_cars_per_capita_elasticity, FE_per_capita_income_elasticity, FE_gas_price_elasticity, FE_cars_per_capita_elasticity, Ame_cars_per_capita_elasticity, Ame_gas_price_elasticity, Ame_per_capita_income_elasticity, SA_per_capita_income_elasticity, SA_gas_price_elasticity, SA_cars_per_capita_elasticity)

* Create a Table of the Elasticities
matrix table = J(4, 3, .)
matrix table[1, 1] = elasticity[1, 1]
matrix table[1, 2] = elasticity[1, 2]
matrix table[1, 3] = elasticity[1, 3]
matrix table[2, 1] = elasticity[1, 4]
matrix table[2, 2] = elasticity[1, 5]
matrix table[2, 3] = elasticity[1, 6]
matrix table[3, 1] = elasticity[1, 7]
matrix table[3, 2] = elasticity[1, 8]
matrix table[3, 3] = elasticity[1, 9]
matrix table[4, 1] = elasticity[1, 10]
matrix table[4, 2] = elasticity[1, 11]
matrix table[4, 3] = elasticity[1, 12]

* Add Row and Column Labels
matrix rownames table = "OLS" "Within" "Amemiya" "Swamy&Arora"
matrix colnames table = "Y/N" "Pmg/Pgdp" "CAR/N"

* Display the Matrix Table
matrix list table

*Save the Table
log using "tablodört.log", replace
matrix list table
log close

*-----------------------------------------------------------------------------
*PART V: TESTS for MODEL I and MODEL II

*Roy-Zellner Test for Poolability (Panel Data OLS coefficients vs. Random Effect Model I [Amemiya] coefficients)
*Create Panel Data OLS (Unpooled OLS) for MODEL I
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP
eststo UnpooledOLS_M1
scalar b1z = _b[LINCOMEP]
scalar b2z = _b[LRPMG]
scalar b3z = _b[LCARPCAP]
scalar constantz = _b[_cons]
*The R-Z Test for MODEL I
test(b1z=b1d)(b2z=b2d)(b3z=b3d)
*reject poolability as p<0.05 (roughly) for MODEL I
*Roy-Zellner Test for Poolability (Panel Data OLS coefficients vs. Random Effect Model II [Amemiya] coefficients)
*Create Panel Data OLS (Unpooled OLS) for MODEL II
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR
eststo UnpooledOLS_M2
scalar b1zz = _b[LINCOMEP]
scalar b2zz = _b[LRPMG]
scalar b3zz = _b[LCARPCAP]
scalar lambdazz = _b[L.LGASPCAR]
scalar constantzz = _b[_cons]
*The R-Z Test for MODEL II
test(b1zz=b1cc)(b2zz=b2cc) (b3zz=b3cc) (lambdazz=lambdacc)  
*reject poolability as p<0.05 (roughly) for MODEL II

*B-P LM Test for the Error Components Models of MODEL I and MODEL II 
*B-P LM Test on the Error Components Model (Static: MODEL I)
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP, re
xttest0
*Random Effects Model is appropriate for MODEL I
*B-P LM Test on the Error Components Model (Dynamic: MODEL II)
xtreg LGASPCAR LINCOMEP LRPMG LCARPCAP L.LGASPCAR, re
xttest0
*Random Effects Model is appropriate for MODEL II

*Durbin-Wu-Hausman Specification Test (Static: MODEL I) (RE[Amemiya] vs. FE)
hausman Within_M1 RE_Amemiya_M1
*Durbin-Wu-Hausman Specification Test (Static: MODEL I) (RE[S&A] vs. FE)
hausman Within_M1 RE_SwamyArora_M1
