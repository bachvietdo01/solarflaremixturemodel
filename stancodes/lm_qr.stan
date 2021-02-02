data {
  int<lower=1> N; // Number of data
  int<lower=1> D; // Number of covariates
  matrix[N, D] X;
  real y[N];

  int<lower=1> N_test;
  matrix[N_test, D] X_test;
  real y_test[N_test];
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
  real alpha;           
  vector[D] theta;
  real<lower=0> sigma2;
}

transformed parameters {
  real<lower=0> sigma;

  sigma = sqrt(sigma2);
}

model {
  sigma2 ~ inv_gamma(1, 1);
  alpha ~ normal(0, 1);
  theta ~ normal(0, 1);
  
  y ~ normal(alpha + Q_ast * theta, sigma);
}

generated quantities {
  vector[N_test] y_pred;
  vector[N_test] indiv_squared_errors;

  real <lower = 0> sum_of_squares;
  real <lower = 0> root_mean_squared_error;

  vector[D] beta;
  vector[N_test] mu_new;

  {
  	beta = R_ast_inverse * theta;
    mu_new = alpha + X_test * beta;
    
    for (n in 1:N_test)  {
      y_pred[n] = normal_rng(mu_new[n], sigma);
      indiv_squared_errors[n] = (y_test[n] - y_pred[n])^2;
    }
  }
  // sum of squares, data set size and scale specific
  sum_of_squares = sum(indiv_squared_errors);

  // divide by number of new / test points and sqrt for RMSE
  root_mean_squared_error = sqrt(sum_of_squares / N_test);
  
}
