data {
  // Training data
  int<lower=1> N; // Number of data
  int<lower=1> D; // Number of covariates
  int<lower=1> K; // Number of mixture components
  
  matrix[N, D] X;
  real y[N];

  // Test data
  int<lower=1> N_test;
  matrix[N_test, D] X_test;
  real y_test[N_test];

  // Constant
  vector[K] dir_alpha; // Dirichlet params
}

// this step does some transformations to the data
transformed data {
  matrix[N, D] Q_ast;
  matrix[D, D] R_ast;
  matrix[D, D] R_ast_inverse;
  
  // thin and scale the QR decomposition
  Q_ast = qr_Q(X)[, 1:D] * sqrt(N - 1);
  R_ast = qr_R(X)[1:D, ] / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);
}

parameters {       
  real<lower=0> sigma;

  simplex[K] pi_uo;

  real alpha_uo[K];                 
  vector[D] theta_uo[K]; 
}

transformed parameters {
// Ordered parameters with a single posterior mode
  simplex[K] pi;
  vector[D] theta[K];
  real alpha[K];

  // 1st cordinates of K mu(s)
  row_vector[K] theta_1d;

  for(k in 1: K)
    theta_1d[k] = theta_uo[k][1];

  // Impose a constraint on model parameters 
  // by the sorting order of mu(s) 1st coordinates
  pi = pi_uo[sort_indices_asc(theta_1d)];
  theta = theta_uo[sort_indices_asc(theta_1d)];
  alpha = alpha_uo[sort_indices_asc(theta_1d)];
}

model {
  real log_ll[K];

  pi_uo ~ dirichlet(dir_alpha);

  for(k in 1:K) {
    alpha_uo[k] ~ normal(0, 1);
    theta_uo[k] ~ normal(0, 1);
  }

  // Data likelihood
  for (n in 1:N){
    for (k in 1:K){
      // increment log probability of the gaussian
      log_ll[k] = log(pi_uo[k]) + normal_lpdf(y[n] | alpha_uo[k] + Q_ast * theta_uo[k], sigma);
    }

    target += log_sum_exp(log_ll);
  }
}

generated quantities {
  vector[N_test] y_pred;
  vector[N_test] indiv_squared_errors;

  real <lower = 0> sum_of_squares;
  real <lower = 0> root_mean_squared_error;

  vector[D] beta;
  real mu_new;
  int z;

  {    
    for (n in 1:N_test)  {
      // get the right cluster pars
      z = categorical_rng(pi);
      beta = R_ast_inverse * theta[z];

      mu_new = alpha[z] + X_test[n,] * beta;
      y_pred[n] = normal_rng(mu_new, sigma);
      indiv_squared_errors[n] = (y_test[n] - y_pred[n])^2;
    }
  }
  // sum of squares, data set size and scale specific
  sum_of_squares = sum(indiv_squared_errors);

  // divide by number of new / test points and sqrt for RMSE
  root_mean_squared_error = sqrt(sum_of_squares / N_test);
}
