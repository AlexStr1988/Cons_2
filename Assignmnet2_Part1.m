%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
rng(5);
%setting up parameters

beta = 0.96;
gamma = 1;
r = 0.04;    
rho = 0.9;    
sigma = 0.15;
w_bar = 2.5;
phi = 2;

% steady-state
n_ss = 40/168;
omega = w_bar / ( n_ss^(1/phi) ); % - based on the partial derivatives of the utility function


%Setting values for rows and col:

row_dim = 600;
col_dim = 5;

%setting up V0 matrix

V0 = zeros(row_dim,col_dim);


%Doing Tauchen function

[y, P] = Tauchen(col_dim, 0, rho, sigma, 2);

%wage
wage = exp(y); % exp of y for wage
w_min = min(wage);

% Labor supply 
n_supply = ( wage ./ omega ).^phi; % - rearrangin omega equation

%setting up a_min and a_max

a_min = - ( w_min * n_ss ) / r; % now y is represented through wages and hours worked
%a_min = min(y(1))/r;
%a_min = (-1)*y(1)/r;
a_max = 10;

a_grid = linspace(a_min, a_max, row_dim);

%Other matrices

V1 = ones(row_dim,col_dim);
policy = zeros(row_dim,col_dim);
consumption = zeros(row_dim,col_dim);

%Setting up tolerance:

e = 1e-9;
diff = 1;
it = 0;
max_it = 10000;

%determining disutility

dis_util = omega .* ( n_supply.^(1 + 1/phi) ) ./ (1 + 1/phi); % just taken from the utility function

% VAlue function:

while diff > e %&& it < max_it
    it = it + 1;

    for j = 1:col_dim

        %Expected value
        EV = V0 * P(j,:)';    % 30x1


        for i = 1:row_dim
            c = a_grid(i) + wage(j) * n_supply(j) - a_grid/(1+r);

            u_cn = -Inf(size(c));
            pos_util = (c - dis_util(j)) > 0;   %positive utilicty when consumption bigger than disutility
            u_cn(pos_util) = log( c(pos_util) - dis_util(j) );

            value = u_cn + beta * EV';   % 1x30

            [V1(i,j), best_a] = max(value);
            policy(i,j) = a_grid(best_a);
            consumption(i,j) = c(best_a);
        end
    end
diff = max(max(abs(V1 -V0)));
V0=V1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Graphs for the Valu Function for each a
figure;
hold on;

colors = lines(col_dim);  % different color for each income state

for j = 1:col_dim
    plot(a_grid, V1(:, j), 'Color', colors(j,:));
end



xlabel('Assets');
ylabel('Value function (a,w)');
title('Value Function for all wage');

grid on;
hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Part C

T = 1000;
forget = 500;  % drop 500

eps = randn(T,1);   % normal shocks

sim_y = zeros(T,1);         
sim_a = zeros(T,1);         
sim_c = zeros(T,1);
sim_n = zeros(T,1);
sim_w = exp(sim_y); %based on the income

% Income AR(1)
for t = 2:T
    sim_w(t) =(1- rho)*w_bar + rho * sim_w(t-1) + eps(t);
end

% Drop first 500
sim_y = sim_y(forget+1:T);
sim_a = sim_a(forget+1:T);
sim_c = sim_c(forget+1:T);
sim_n = sim_n(forget+1:T);
sim_w = sim_w(forget+1:T);

T_sim = length(sim_w);

% Extracting y difference
for t = 1:T_sim
    [y_period, w_dif(t)] = min(abs(sim_w(t) - wage));
end

% Simulate assets and consumption using policy functions
for t = 1:T_sim-1
    % Assets
    [a_period, a_dif] = min(abs(sim_a(t) - a_grid));
    
    
    % Apply policy function
    sim_a(t+1) = policy(a_dif, w_dif(t));
    sim_c(t) = consumption(a_dif, w_dif(t));
    sim_n(t) = n_supply(w_dif(t));
end

%%% Heatmap
data = [sim_w, sim_a, sim_c,sim_n, sim_y ];

figure;
tiledlayout(5,1);
nexttile; plot (sim_y'); colorbar; title('Y');
nexttile; plot (sim_w'); colorbar; title('W');
nexttile; plot (sim_a'); colorbar; title('A');
nexttile; plot (sim_c'); colorbar; title('C');
nexttile; plot (sim_n'); colorbar; title('N');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Part D

%Calculating standard deviation
std_n = std(sim_n);  %Matlab Function

%SD Code:
N = length(sim_n);           % number of observations
mean_n = sum(sim_n) / N;     % mean

sq_diff_n = (sim_n - mean_n).^2;

variance_n = sum(sq_diff_n) / (N - 1);

std_n_calc = variance_n^(1/2);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Part D responses

%a) Increase

%b) Potentially no change

%c) Increase

%d) Increase

