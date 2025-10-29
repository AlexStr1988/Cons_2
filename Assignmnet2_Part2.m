%%%%%%Assignment 2 - Part 2

%Setting up parameters

beta = 0.96; 
alpha = 0.4; 
delta0 = 0.1; 
phi2 = 0.2; 
phi1 = 1/beta - (1-delta0);
w_bar = 2.5;

rho = 0.9; 
sigma = 0.15;

%Pre-determining the size
col_dim = 5;
row_dim = 3000;

% Capital limits
k_min = 0; %similar to borrowing constraint but in this case 0 is the natural limit
k_max = 10;

%Similar to a_grid
k_grid = linspace(k_min, k_max, row_dim);

%Doing the same as in PART1 for wages
[y, P] = Tauchen(col_dim, 0, rho, sigma, 2);
wage = exp(y);

% Setting up the initial matrices
V0 = zeros(row_dim, col_dim);
V1 = ones(row_dim, col_dim);

policy_k = zeros(row_dim, col_dim);
policy_n = zeros(row_dim, col_dim);
policy_u = zeros(row_dim, col_dim);
policy_inv = zeros(row_dim, col_dim);

%Setting up tolerance:

e = 1e-9;
diff = 1;
it = 0;
max_it = 2;

%Getting the VAlue Function

while diff > e && it < max_it
    it = it + 1;
    
    for j = 1:col_dim
        EV = V0 * P(j,:)'; % expected value
        
        for i = 1:row_dim
            k_prime = k_grid;
       

            %Based on the FOC with respect to u we have the following u
            %equation
            u = 1 + ( alpha * (1-alpha)^((1-alpha)/alpha) * wage(j)^(-(1-alpha)/alpha) - phi1 ) / phi2;


            %Based on the FOC with respect to n we have the following n
            %equation
            n = ((1 - alpha) * (u * k_grid(i))^alpha / wage(j))^(1 / alpha);


             % Calculating depreciation
             delta_u = delta0 + phi1 * (u - 1) + phi2/2 * (u - 1)^2;

             % Calculating investement based on the previous variables
             inv = k_prime - (1 - delta_u) * k_grid(i);

             %Taking only positive investements
             %pos_inv = inv >= 0;

             %Calculating production
             prod = (u*k_grid(i))^alpha * n^(1-alpha) - wage(j)*n - inv;

             %Calculating value function for positive investements
             value = prod + beta * EV';
           
            % Updating optimal values
            [V1(i,j), best] = max(value);
            policy_k(i,j) = k_prime(best);
            policy_u(i,j) = u;
            policy_n(i,j) = n;
            policy_inv(i,j) = inv(best);
        end
    end
    
    % Convergence
    diff = max(max(abs(V1 - V0)));
    V0 = V1;
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PART B

figure;
hold on;

colors = lines(col_dim); 

for j = 1:col_dim
    plot(k_grid, V1(:, j), 'Color', colors(j,:));
end

xlabel('Capital k');
ylabel('Value function V(k,w)');
title('Value Function for all wage states');

grid on;
hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Part C

T = 1000;
forget = 500;   % drop 500 periods

eps =  randn(T,1);  % normal shcoks

sim_w = zeros(T,1);
sim_k = zeros(T,1);
sim_n = zeros(T,1);
sim_inv = zeros(T,1);
sim_u = zeros(T,1);

% Initializing the starting minimal values
sim_k(1) = k_min;         
sim_w(1) = wage(1);  

% Simulate income as AR(1)
for t = 2:T
    sim_w(t) = (1 - rho)*w_bar + rho*sim_w(t-1) + eps(t);
end

% Drop first 500 periods
sim_w = sim_w(forget+1:T);
sim_k = sim_k(forget+1:T);
sim_n = sim_n(forget+1:T);
sim_inv = sim_inv(forget+1:T);
sim_u = sim_u(forget+1:T);
T_sim = length(sim_w);

%Extracting wage difference
for t = 1:T_sim
    [w_period, w_dif(t)] = min(abs(sim_w(t) - wage));
end

% Simulate other variables
for t = 1:T_sim-1

    % Find k difference
    [k_period, k_dif_i] = min(abs(sim_k(t) - k_grid));

    % Apply policy functions
    sim_k(t+1)   = policy_k(k_dif_i,w_dif(t));
    sim_n(t)     = policy_n(k_dif_i,w_dif(t));
    sim_u(t)     = policy_u(k_dif_i,w_dif(t));
    sim_inv(t)   = policy_inv(k_dif_i,w_dif(t));
end
           
%%%%Tiled Layout

%%% Heatmap
data = [sim_k, sim_n, sim_inv, sim_u, sim_w ];

figure;
tiledlayout(5,1);
nexttile; plot (sim_k'); colorbar; title('K');
nexttile; plot (sim_w'); colorbar; title('W');
nexttile; plot (sim_inv'); colorbar; title('INV');
nexttile; plot (sim_u'); colorbar; title('U');
nexttile; plot (sim_n'); colorbar; title('N');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Part D

%Calculating standard deviation
std_n = std(sim_n);  %Matlab Function

%a) Decreases 

%b) Decreases

