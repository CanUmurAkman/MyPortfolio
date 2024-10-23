#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# Specify the directory path you want to set as the working directory
directory_path = "/Users/macbookpro/Desktop"

# Change the working directory
os.chdir(directory_path)

# Verify that the working directory has been changed
print("Working directory has been set to:", directory_path)


# # Preliminaries

# In[2]:


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(123)


# In[3]:


# Parameters
T = 100  # Number of observations
K = 2    # Number of variables
max_lag = 5  # Maximum lag length to consider


# In[4]:


# Define different combinations of AR forms
ar_forms = [
    {"intercept": False, "trend": False},
    {"intercept": True, "trend": False},
    {"intercept": False, "trend": True},
    {"intercept": True, "trend": True}
]

# Define different levels of cross-correlation
cross_correlation_values = [0, 0.3, 0.6, 0.9]


# In[5]:


# Coefficients for the VAR equations (AR(1))
coefficients = np.array([[0.6, -0.4],   # Coefficients for lagged y1
                         [0.3, 0.8]])   # Coefficients for lagged y2


# # DATA GENERATING PROCESS for Reduced Form VAR(1) (for differing levels of cross-correlation (or covariance) and AR forms.)

# In[6]:


# Loop over different cross-correlation values
for correlation in cross_correlation_values:
    print(f"Cross-correlation: {correlation}")
    
    # Generate covariance matrix with the specified cross-correlation
    covariance_matrix = np.array([[1.0, correlation],  
                                    [correlation, 1.0]])

    # Generate VAR errors with cross-correlation
    mean = np.zeros(K)
    errors = np.random.multivariate_normal(mean, covariance_matrix, size=T)
        
    # Compute correlation between the two error series in 'errors'
    correlation_errors = np.corrcoef(errors[:, 0], errors[:, 1])[0, 1]
    print("Correlation between error series in 'errors':", correlation_errors)

    # Loop over different AR forms
    for ar_form in ar_forms:
        intercept = ar_form["intercept"]
        trend = ar_form["trend"]
        print(f"AR Form: Intercept={intercept}, Trend={trend}")

        # Simulate SVAR data for p=1
        data = np.zeros((T, K))
        for t in range(1, T):
            lagged_values = data[t-1:t, :]  # Lagged values of the variables

            # Compute the SVAR equation for each variable
            for i in range(K):
                lagged_product = np.dot(lagged_values.flatten(), coefficients[:, i])  # Compute lagged product
                intercept_term = 21 if intercept else 0
                trend_term = 0.01 * t if trend else 0
                data[t, i] = lagged_product + intercept_term + trend_term + errors[t, i]

        # Compute AIC for p = 1, 2, 3, 4, 5
        aic_values = []
        for p in range(1, max_lag + 1):
            model = sm.tsa.VAR(data)
            results = model.fit(p)
            aic_values.append(results.aic)

        # Print AIC values for different lag lengths
        print("AIC values:", aic_values)   


# # DATA GENERATING PROCESS for  ARFVAR(1) (for differing levels of cross-correlation (or covariance) and AR forms.)

# In[7]:


# Loop over different cross-correlation values
for correlation in cross_correlation_values:
    print(f"Cross-correlation: {correlation}")
    
    # Generate covariance matrix with the specified cross-correlation
    covariance_matrix = np.array([[1.0, correlation],  
                                    [correlation, 1.0]])
    
    # Generate VAR errors with cross-correlation
    mean = np.zeros(K)
    errors = np.random.multivariate_normal(mean, covariance_matrix, size=T)

    # Perform Cholesky decomposition
    chol_decomp = np.linalg.cholesky(covariance_matrix)
    
    # Print the Lower Triangular Matrix B
    print("Lower Triangular Matrix B:")
    print(np.array2string(chol_decomp, separator=', ', formatter={'float_kind':lambda x: "%.2f" % x}))
    
    # Transform errors to make them orthogonal
    orthogonal_errors = np.dot(errors, np.linalg.inv(chol_decomp.T))
    
    # Compute correlation between the two error series in 'orthogonal_errors'
    correlation_orthogonal_errors = np.corrcoef(orthogonal_errors[:, 0], orthogonal_errors[:, 1])[0, 1]
    print("Correlation between error series in 'orthogonal_errors':", correlation_orthogonal_errors)

    # Loop over different AR forms
    for ar_form in ar_forms:
        intercept = ar_form["intercept"]
        trend = ar_form["trend"]
        print(f"AR Form: Intercept={intercept}, Trend={trend}")

        # Simulate SVAR data for p=1
        data = np.zeros((T, K))
        for t in range(1, T):
            lagged_values = data[t-1:t, :]  # Lagged values of the variables

            # Compute the SVAR equation for each variable
            for i in range(K):
                lagged_product = np.dot(lagged_values.flatten(), coefficients[:, i])  # Compute lagged product
                intercept_term = 21 if intercept else 0
                trend_term = 0.01 * t if trend else 0
                data[t, i] = lagged_product + intercept_term + trend_term + orthogonal_errors[t, i]

        # Compute AIC for p = 1, 2, 3, 4, 5
        aic_values = []
        for p in range(1, max_lag + 1):
            model = sm.tsa.VAR(data)
            results = model.fit(p)
            aic_values.append(results.aic)

        # Print AIC values for different lag lengths
        print("AIC values:", aic_values)    


# In[8]:


# Print the Lower Triangular Matrix B
print("Lower Triangular Matrix B:")
print(np.array2string(chol_decomp, separator=', ', formatter={'float_kind':lambda x: "%.2f" % x}))


# ## IRFs of RFVAR (Impulse Responses for each series y1t and y2t)

# In[11]:


from matplotlib.backends.backend_pdf import PdfPages

# Create a PDF file to store all plots
pdf_filename = "RFVAR_IRF_plots.pdf"
pdf_pages = PdfPages(pdf_filename)

# Loop over different cross-correlation values
for correlation in cross_correlation_values:
    print(f"Cross-correlation: {correlation}")
    
    # Loop over different AR forms
    for ar_form in ar_forms:
        intercept = ar_form["intercept"]
        trend = ar_form["trend"]
        print(f"AR Form: Intercept={intercept}, Trend={trend}")

        # Generate covariance matrix with the specified cross-correlation
        covariance_matrix = np.array([[1.0, correlation],  
                                      [correlation, 1.0]])

        # Generate VAR errors with cross-correlation
        mean = np.zeros(K)
        errors = np.random.multivariate_normal(mean, covariance_matrix, size=T)

        # Simulate VAR data for p=1
        data = np.zeros((T, K))
        for t in range(1, T):
            lagged_values = data[t-1:t, :]  # Lagged values of the variables

            # Compute the VAR equation for each variable
            for i in range(K):
                lagged_product = np.dot(lagged_values.flatten(), coefficients[:, i])  # Compute lagged product
                intercept_term = 21 if intercept else 0
                trend_term = 0.01 * t if trend else 0
                data[t, i] = lagged_product + intercept_term + trend_term + errors[t, i]

        # Fit VAR model for p = 1
        model = sm.tsa.VAR(data)
        results = model.fit(1)

        # Compute impulse response functions
        irf = results.irf(10)  # Compute IRFs for 10 periods
        fig = irf.plot(orth=False)   # Plot IRFs without orthogonalization

        # Set x-axis ticks for all subplots
        for i in range(len(fig.axes)):
            fig.axes[i].set_xticks(np.arange(0, 11, 2))

        # Set a customized title for all plots
        fig.suptitle(f"Impulse Response for Cross-correlation: {correlation}, Intercept: {intercept}, Time Trend: {trend}")

        # Update plot titles to denote the shock/error term
        for ax in fig.get_axes():
            title = ax.get_title()
            if "->" in title:
                title_parts = title.split("->")
                title_parts[0] = "ε" + title_parts[0].strip()
                ax.set_title(" -> ".join(title_parts))

        # Export the figure to the PDF file
        pdf_pages.savefig(fig)

        # Show the plots
        plt.close(fig)  # Close the current figure

# Close the PDF file
pdf_pages.close()

print("Plots exported to PDF:", pdf_filename)


# ## IRFs of ARFVAR (Impulse Responses for each series y1t and y2t)

# In[12]:


from matplotlib.backends.backend_pdf import PdfPages

# Create a PDF file to store all plots
pdf_filename = "ARFVAR_IRF_plots.pdf"
pdf_pages = PdfPages(pdf_filename)

# Loop over different cross-correlation values
for correlation in cross_correlation_values:
    print(f"Cross-correlation: {correlation}")
    
    # Loop over different AR forms
    for ar_form in ar_forms:
        intercept = ar_form["intercept"]
        trend = ar_form["trend"]
        print(f"AR Form: Intercept={intercept}, Trend={trend}")

        # Generate covariance matrix with the specified cross-correlation
        covariance_matrix = np.array([[1.0, correlation],  
                                      [correlation, 1.0]])

        # Generate VAR errors with cross-correlation
        mean = np.zeros(K)
        errors = np.random.multivariate_normal(mean, covariance_matrix, size=T)

        # Perform Cholesky decomposition
        chol_decomp = np.linalg.cholesky(covariance_matrix)

        # Transform errors to make them orthogonal
        orthogonal_errors = np.dot(errors, np.linalg.inv(chol_decomp.T))

        # Simulate SVAR data for p=1
        data = np.zeros((T, K))
        for t in range(1, T):
            lagged_values = data[t-1:t, :]  # Lagged values of the variables

            # Compute the SVAR equation for each variable
            for i in range(K):
                lagged_product = np.dot(lagged_values.flatten(), coefficients[:, i])  # Compute lagged product
                intercept_term = 21 if intercept else 0
                trend_term = 0.01 * t if trend else 0
                data[t, i] = lagged_product + intercept_term + trend_term + orthogonal_errors[t, i]

        # Fit SVAR model for p = 1
        model = sm.tsa.VAR(data)
        results = model.fit(1)

        # Compute impulse response functions
        irf = results.irf(10)  # Compute IRFs for 10 periods
        fig = irf.plot(orth=False)   # Plot IRFs without orthogonalization

        # Set x-axis ticks for all subplots
        for i in range(len(fig.axes)):
            fig.axes[i].set_xticks(np.arange(0, 11, 2))

        # Set a customized title for all plots
        fig.suptitle(f"Impulse Response for Cross-correlation: {correlation}, Intercept: {intercept}, Time Trend: {trend}")

        # Update plot titles to denote the shock/error term
        for ax in fig.get_axes():
            title = ax.get_title()
            if "->" in title:
                title_parts = title.split("->")
                title_parts[0] = "ε" + title_parts[0].strip()
                ax.set_title(" -> ".join(title_parts))

        # Export the figure to the PDF file
        pdf_pages.savefig(fig)

        # Show the plots
        plt.close(fig)  # Close the current figure

# Close the PDF file
pdf_pages.close()

print("Plots exported to PDF:", pdf_filename)

