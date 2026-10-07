# Live Demo

## Scenario 1: The Ideal Borrower (Model APPROVES)

### Goal
Demonstrate how strong financial capability ratios (high income, low loan ratio, long job tenure) result in a clear **APPROVE** decision.

### UI Form Inputs
* **Annual income:** `250,000`
* **Loan amount:** `300,000`
* **Annual repayment (annuity):** `15,000`
* **Price of goods financed:** `270,000`
* **Age:** `42`
* **Currently employed:** `Checked`
* **Years in current job:** `10.0`
* **Children:** `0`
* **Family members:** `2`
* **Education:** `Higher education`
* **Income type:** `State servant`
* **Family status:** `Married`
* **Housing:** `House / apartment`
* **Occupation:** `Managers` *(or leave blank)*
* **Loan type:** `Cash loans`
* **Owns a car:** `Y`
* **Car age:** `3`
* **Owns property:** `Y`

## Scenario 2: High-Risk / Over-Leveraged Applicant (Model REJECTS)

### Goal
Show how excessive debt ratios relative to income and low job stability trigger an instant **REJECT** decision—even for a thin-file applicant with no credit history.

### UI Form Inputs
* **Annual income:** `50,000`
* **Loan amount:** `1,000,000`
* **Annual repayment (annuity):** `150,000`
* **Price of goods financed:** `500,000`
* **Age:** `22`
* **Currently employed:** `Checked`
* **Years in current job:** `0.5`
* **Children:** `0`
* **Family members:** `1`
* **Education:** `Secondary / secondary special`
* **Income type:** `Working`
* **Family status:** `Single / not married`
* **Housing:** `With parents`
* **Occupation:** `Laborers` *(or leave blank)*
* **Loan type:** `Cash loans`
* **Owns a car:** `N`
* **Car age:** `0`
* **Owns property:** `N`

## Scenario 3: Borderline Case & Decision Threshold Sweep

### Goal
Demonstrate how the calibrated probability output allows bank executives to dynamically adjust risk tolerance ($R$) based on business policy.

### UI Form Inputs
* **Annual income:** `120,000`
* **Loan amount:** `450,000`
* **Annual repayment (annuity):** `28,000`
* **Price of goods financed:** `400,000`
* **Age:** `28`
* **Currently employed:** `Checked`
* **Years in current job:** `2.0`
* **Children:** `1`
* **Family members:** `3`
* **Education:** `Secondary / secondary special`
* **Income type:** `Working`
* **Family status:** `Married`
* **Housing:** `House / apartment`
* **Occupation:** *(leave blank)*
* **Loan type:** `Cash loans`
* **Owns a car:** `N`
* **Car age:** `0`
* **Owns property:** `Y`
