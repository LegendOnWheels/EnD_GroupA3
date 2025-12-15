%% Load data
clc; clear;
load Data.mat
fs = 16000;

%% Data preprocessing
% STFT
frame_len = round(0.02 * fs);  % 20 ms = 320 samples
hop = frame_len/2;             % 160 samples
win = hann(frame_len, 'periodic');
NFFT = frame_len*4; % Zero padding by frame_len*3 - improves freq resolution

% Preallocate
[S_tmp, ~, ~] = stft(Data(:,1), fs, 'Window', win, ...
    'OverlapLength', frame_len - hop, 'FFTLength', NFFT);

[F, L] = size(S_tmp);
Ymat = zeros(F, L, nrmics);

% Now fill the array
for m = 1:nrmics
    Ymat(:,:,m) = stft(Data(:,m), fs, 'Window', win, ...
        'OverlapLength', frame_len - hop, 'FFTLength', NFFT);
end


% Noise variance estimation (use first 1 second, noise-only)

noise = fs * 1;   % first 1 second

varw2 = zeros(F, nrmics);

for m = 1:nrmics
    [S_noise, ~, ~] = stft(Data(1:noise,m), fs, ...
        'Window', win, ...
        'OverlapLength', frame_len - hop, ...
        'FFTLength', NFFT);

    % noise variance per frequency
    varw2(:,m) = mean(abs(S_noise).^2, 2);
end

%% Task 1 - MLE estimator

S_hat = zeros(F, L);

for f = 1:F
    for l = 1:L
        num = 0;
        den = 0;
        for m = 1:nrmics
            num = num + Ymat(f,l,m) / varw2(f,m);
            den = den + 1 / varw2(f,m);
        end
        S_hat(f,l) = num / den;
    end
end

% Reconstruct enhanced speech
estimator = istft(S_hat, fs, ...
    'Window', win, ...
    'OverlapLength', frame_len - hop, ...
    'FFTLength', NFFT);

%% Task 2 - Variance of estimator
% Clean speech STFT (same parameters as noisy)
[S_clean, ~, ~] = stft(Clean, fs, ...
    'Window', win, ...
    'OverlapLength', frame_len - hop, ...
    'FFTLength', NFFT);

% Precompute weights 1/sigma^2
W = 1 ./ varw2;   % size [F x M]

var_emp = zeros(nrmics,1);

for N = 1:nrmics
    
    % weights for the first N mics
    WN = W(:,1:N);              % [F x N]
    den = sum(WN, 2);           % [F x 1]

    % reshape to broadcast across frames
    WN_3D = reshape(WN, F, 1, N);  % [F x 1 x N]

    % numerator: sum_m Y(:,:,m) * W(f,m)
    num = sum(Ymat(:,:,1:N) .* WN_3D, 3); % [F x L]

    % S_hat
    S_hat_N = num ./ den;     % [F x L]

    % empirical variance
    diff = S_hat_N - S_clean;
    var_emp(N) = mean(abs(diff(:)).^2);

end

% plot
figure;
plot(1:nrmics, var_emp, '-o');
xlabel('Number of microphones');
ylabel('Empirical variance');
title('Task 2: Empirical variance vs number of microphones');
grid on;

%% Task 3: CRLB per frequency and averaged
F = size(varw2,1);

crlb_avg = zeros(nrmics,1);  % store CRLB averaged over frequency

for N = 1:nrmics
    
    % CRLB(f) = 1 / sum_{m=1..N} 1/varw2(f,m)
    crlb_f = 1 ./ sum(1 ./ varw2(:,1:N), 2);   % F x 1 vector
    
    % average across all frequency bins
    crlb_avg(N) = mean(crlb_f);

end

figure;
plot(1:nrmics, crlb_avg, '-o');
xlabel('Number of microphones');
ylabel('CRLB (averaged over frequency)');
title('Task 3: CRLB vs number of microphones');
grid on;

% Comparison
figure;
plot(1:nrmics, var_emp, '-o'); hold on;
plot(1:nrmics, crlb_avg, '-o');
legend('Empirical variance','CRLB');
xlabel('Number of microphones');
ylabel('Variance');
title('Variance vs CRLB');
grid on;

%% Spectrograms: Noisy, Clean, Enhanced

% --- STFT of Noisy mic 1 (for plotting)
[S_noisy, Fv, Tv] = stft(Data(:,1), fs, ...
    'Window', win, ...
    'OverlapLength', frame_len-hop, ...
    'FFTLength', NFFT);

% --- STFT of Clean speech (ground truth)
[S_clean, ~, ~] = stft(Clean, fs, ...
    'Window', win, ...
    'OverlapLength', frame_len-hop, ...
    'FFTLength', NFFT);

figure;

% Noisy
subplot(3,1,1);
imagesc(Tv, Fv, 20*log10(abs(S_noisy) + 1e-6));
axis xy;
colormap jet;
colorbar;
title('Noisy Speech (Mic 1)');
ylabel('Frequency [Hz]');

% Clean
subplot(3,1,2);
imagesc(Tv, Fv, 20*log10(abs(S_clean) + 1e-6));
axis xy;
colormap jet;
colorbar;
title('Clean Speech');
ylabel('Frequency [Hz]');

% MLE
subplot(3,1,3);
imagesc(Tv, Fv, 20*log10(abs(S_hat) + 1e-6));
axis xy;
colormap jet;
colorbar;
title('Enhanced Speech (MLE)');
ylabel('Frequency [Hz]');
xlabel('Time [s]');
