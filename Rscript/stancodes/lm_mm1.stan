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
  matrix[D, D] I; // Identity matrix
  vector[K] dir_alpha; // Dirichlet params
}

parameters {
  real alpha;           
  vector[D] beta;
  real<lower=0> sigma;

  simplex[K] pi_uo;             
  vector[D] mu_uo[K]; 
}

transformed parameters {
// Ordered parameters with a single posterior mode
  simplex[K] pi;
  vector[D] mu[K];

  // 1st cordinates of K mu(s)
  row_vector[K] mu_1d;

  for(k in 1: K)
    mu_1d[k] = mu_uo[k][1];

    // Impose a constraint on model parameters 
    // by the sorting order of mu(s) 1st coordinates
    pi = pi_uo[sort_indices_asc(mu_1d)];
    mu = mu_uo[sort_indices_asc(mu_1d)];
}

model {
  real beta_dist[K];

  alpha ~ normal(0, 1);
  pi_uo ~ dirichlet(dir_alpha);

  for(k in 1:K) {
    mu_uo[k] ~ normal(0, 1);
  }

  //prior of beta is a finite mixture model
  for (k in 1:K){
    // increment log probability of the gaussian
    beta_dist[k] = log(pi_uo[k]) + multi_normal_lpdf(beta| mu_uo[k], I);
  }
  
  target += log_sum_exp(beta_dist);
  
  y ~ normal(alpha + X * beta, sigma);
}

generated quantities {
  vector[N_test] y_pred;
  vector[N_test] indiv_squared_errors;

  real <lower = 0> sum_of_squares;
  real <lower = 0> root_mean_squared_error;

  vector[N_test] mu_new;
  vector[D] beta_new;
  int z;

  {
    z = categorical_rng(pi);
    beta_new = multi_normal_rng(mu[z], I);
    mu_new = alpha + X_test * beta_new;
    
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
