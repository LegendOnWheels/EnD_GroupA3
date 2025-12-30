%% Multi-microphone speech enhancement

%   Y_m(l,k)  -> STFT of mic m at frame l, frequency bin k
%   Y         -> stored as Y(k,l,m) of size [K x L x M]
%   sigmaW2   -> noise variance σ^2_{W_m}(k), stored as [K x M]
%   S_hat     -> estimator Ŝ(l,k), stored as S_hat(k,l)

%% SETUP

%   Load data

clc; clear;
load('Data.mat');   
fs = 16000;
M  = double(nrmics);

%   STFT parameters

frame_len = round(0.02 * fs);  
hop       = frame_len/2;            
win       = hann(frame_len,'periodic');
NFFT      = frame_len * 4;
overlap   = frame_len - hop;

%   STFT for all microphones

[Y1, f_axis, t_axis] = stft(Data(:,1), fs, 'Window', win, ...
    'OverlapLength', overlap, 'FFTLength', NFFT);

[K, L] = size(Y1);              
Y = zeros(K, L, M);
Y(:,:,1) = Y1;

for m = 2:M
    Y(:,:,m) = stft(Data(:,m), fs, 'Window', win, ...
        'OverlapLength', overlap, 'FFTLength', NFFT);
end

%   Estimate noise variance from first 1 second

noise_samples = min(size(Data,1), fs * 1);
sigmaW2 = zeros(K, M);

for m = 1:M
    Y_noise = stft(Data(1:noise_samples, m), fs, 'Window', win, ...
        'OverlapLength', overlap, 'FFTLength', NFFT);
    sigmaW2(:,m) = mean(abs(Y_noise).^2, 2);
end

%   Numerical floor to avoid divide by zero

sigmaW2 = max(sigmaW2, 1e-12);

%% Question 1: Estimator for target signal
% Ŝ(l,k) = [sum_m Y_m(l,k)/σ^2_{W_m}(k)] / [sum_m 1/σ^2_{W_m}(k)]

S_hat = zeros(K, L);

for k = 1:K
    for ell = 1:L
        num = 0;
        den = 0;
        for m = 1:M
            num = num + Y(k, ell, m) / sigmaW2(k, m);
            den = den + 1 / sigmaW2(k, m);
        end
        S_hat(k, ell) = num / den;
    end
end

s_hat = istft(S_hat, fs, 'Window', win, 'OverlapLength', overlap, 'FFTLength', NFFT);

%% Question 2: Empirical variance
% var_emp(N) = (1/(K*L)) * sum_{k,l} |Ŝ_N(l,k) - S(l,k)|^2

S_clean = stft(Clean(:), fs, 'Window', win, 'OverlapLength', overlap, 'FFTLength', NFFT);

var_emp = zeros(M, 1);

for N = 1:M
    S_hat_N = zeros(K, L);

    for k = 1:K
        for ell = 1:L
            num = 0;
            den = 0;
            for m = 1:N
                num = num + Y(k, ell, m) / sigmaW2(k, m);
                den = den + 1 / sigmaW2(k, m);
            end
            S_hat_N(k, ell) = num / den;
        end
    end

    diff = S_hat_N - S_clean;
    var_emp(N) = mean(abs(diff(:)).^2);
end

figure(1);
plot(1:M, var_emp, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Number of microphones used (N)');
ylabel('Empirical variance  var_{emp}');
title('Q2: Empirical variance vs number of microphones');

%% Question 3: CRLB per frequency band and averaged CRLB vs number of microphones
% CRLB(k;N) = 1 / sum_{m=1..N} [1/σ^2_{W_m}(k)]

crlb_k   = zeros(K, M);    
crlb_avg = zeros(M, 1); 

for N = 1:M
    info_k = sum(1 ./ sigmaW2(:, 1:N), 2);  
    crlb_k(:, N) = 1 ./ info_k;
    crlb_avg(N)  = mean(crlb_k(:, N));
end

%   (a) Averaged CRLB as a function of # microphones

figure(2);
plot(1:M, crlb_avg, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Number of microphones used (N)');
ylabel('CRLB (averaged over frequency)');
title('Q3: Averaged CRLB vs number of microphones');

%   (b) Empirical variance vs averaged CRLB as a function of # microphones

figure(3);
plot(1:M, var_emp,  '-o', 'LineWidth', 1.5); hold on;
plot(1:M, crlb_avg, '-s', 'LineWidth', 1.5);
grid on;
xlabel('Number of microphones used (N)');
ylabel('Variance / bound');
title('Q3: Empirical variance vs averaged CRLB');
legend('Empirical var_{emp}', 'Averaged CRLB', 'Location', 'northeast');

%   (c) CRLB per frequency band as a function of # microphones

N_show = unique([1 2 4 8 M]);
N_show = N_show(N_show <= M);

figure(4); hold on;
for i = 1:numel(N_show)
    N = N_show(i);
    plot(f_axis, crlb_k(:, N), 'LineWidth', 1.2);
end
grid on;
xlabel('Frequency (Hz)');
ylabel('CRLB(k)');
title('Q3: CRLB per frequency band (selected N)');
legend(arrayfun(@(n) sprintf('N=%d', n), N_show, 'UniformOutput', false), ...
    'Location', 'northeast');

%   (d) Numeric comparison (Does the estimator reach the CRLB?)

ratio   = var_emp ./ crlb_avg;
rel_gap = (var_emp - crlb_avg) ./ crlb_avg;

fprintf('\nQ3: Empirical variance vs averaged CRLB\n');
fprintf('  N     var_emp         crlb_avg        ratio(var/crlb)   rel_gap\n');
for N = 1:M
    fprintf('%3d  %12.4e   %12.4e     %10.4f        %10.4f\n', ...
        N, var_emp(N), crlb_avg(N), ratio(N), rel_gap(N));
end
fprintf('Best ratio (closest to 1):  N=%d, ratio=%.4f\n', find(ratio==min(ratio),1), min(ratio));
fprintf('Worst ratio:                N=%d, ratio=%.4f\n\n', find(ratio==max(ratio),1), max(ratio));

%% Spectrogram comparison

[Y_noisy1, Fv, Tv] = stft(Data(:,1), fs, 'Window', win, 'OverlapLength', overlap, 'FFTLength', NFFT);

figure(5);
subplot(3,1,1);
imagesc(Tv, Fv, 20*log10(abs(Y_noisy1) + 1e-8));
axis xy; colormap jet; colorbar;
title('Noisy speech (Mic 1)');
ylabel('Frequency (Hz)');

subplot(3,1,2);
imagesc(Tv, Fv, 20*log10(abs(S_clean) + 1e-8));
axis xy; colormap jet; colorbar;
title('Clean speech (reference)');
ylabel('Frequency (Hz)');

subplot(3,1,3);
imagesc(Tv, Fv, 20*log10(abs(S_hat) + 1e-8));
axis xy; colormap jet; colorbar;
title('Enhanced speech (ML/GLS, all microphones)');
ylabel('Frequency (Hz)');
xlabel('Time (s)');

%% Audio listening
%soundsc(Data(:,1), fs); pause(length(Data(:,1))/fs + 0.25);
%soundsc(s_hat, fs);
