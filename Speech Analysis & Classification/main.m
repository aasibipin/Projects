%% Part A

load("lab4data.mat");
speech = FEMALES;
fs = 12500;
tf = [(1:length(speech))]/fs;
N = length(speech);
M = N - 1; 

% Manual Autocorr 
total = 0;
ACF_speech = zeros(1,length(speech));
for k = 1:M
    for n = 1:N-k+1
        total = total + speech(n).*speech(n+k-1);
    end
    ACF_speech(k) = total;
    total = 0;
end
ACF_speech= ACF_speech./max(ACF_speech);

% Plotting Female Signal Time Domain
figure
plot(tf,speech)
title('Female Speech Time Domain')
xlabel('Time (s)')
ylabel('Amplitude')

% Plotting Autocorr & Finding Pitch Peak Period
NUMBER_PEAKS = 4;
figure
plot(tf,ACF_speech)
[pks, locs] = findpeaks(ACF_speech,tf,'MinPeakDistance',0.003,'NPeaks', NUMBER_PEAKS);
findpeaks(ACF_speech,tf,'MinPeakDistance',0.003,'SortStr','descend','NPeaks', NUMBER_PEAKS);
title('Female Autocorrelation Pitch Detection')
xlabel('Time Lag (s)')
ylabel('Coefficiant of Correlation')

dist = 0;
for i = (1:NUMBER_PEAKS-1)
    dist(i) = locs(i+1)-locs(i);
end

pitch_period = mean(dist)


% Plot of the spectrum 
speech_FFT = abs(fft(speech));
ff_bins = 0:N-1;
ff_hz = ff_bins*fs/N;
figure
plot(ff_hz(1:M/2), speech_FFT(1:M/2));
xlabel('Frequency (Hz)');
ylabel('Magnitude Spectrum');
title('Speech Spectrum of Female Voice');
axis tight; 
% Finding all peaks in signal
Select_FFT = speech_FFT(1:M/2);
[f_pks, f_locs] = findpeaks(Select_FFT,ff_hz(1:M/2),'MinPeakDistance',80,'MinPeakProminence',2);
hold on
plot(f_locs,f_pks,'o')
% Fitting a polynomial line with order = 7 to the minimums of line
p = polyfit(f_locs, -f_pks',7);     % f_pks is negative to find peaks
v = polyval(p, f_locs);
plot(f_locs,v)
% Finding x-axis boundaries of formants by finding peaks of Polynomial Line
[seg_pks, seg_locs] = findpeaks(v,f_locs,'MinPeakDistance',200);
plot (seg_locs,seg_pks,'o') 


% Creating Formant Arrays containing peak and location
formants_pks = 0;
formants_locs = 0;
for n = 1:length(seg_locs)
    for i = 1:length(f_locs)
        if (n == 1) && (f_locs(i) < seg_locs(n))
            formants_pks(i,n) = f_pks(i);
            formants_locs(i,n) = f_locs(i);
        elseif (n == length(seg_locs)) && (f_locs(i) > seg_locs(n))
            formants_pks(i,n+1) = f_pks(i);
            formants_locs(i,n+1) = f_locs(i);
        elseif (f_locs(i) ==  seg_locs(n))
            formants_pks(i,n) = f_pks(i);
            formants_locs(i,n) = f_locs(i);      
        elseif (f_locs(i) <  seg_locs(n)) && (f_locs(i) > seg_locs(n-1))
            formants_pks(i,n) = f_pks(i);
            formants_locs(i,n) = f_locs(i);
        end
    end
end


% Creating Rectanlges around each formant
for n = 1:length(formants_pks(1,:))
    if (n==1)
        rectangle('Position', [0 0 seg_locs(1,n) max(formants_pks(:,n))+4], ...
            'EdgeColor', 'b', 'LineWidth',2, 'LineStyle','-.')
    elseif (n == length(formants_pks(1,:)))
        rectangle ('Position', [formants_locs(find(formants_locs(:,n),1),n)-100 0 ...
            (max(formants_locs(:,n))-formants_locs(find(formants_locs(:,n),1),n)+200) ...
            max(formants_pks(:,n))+4], 'EdgeColor', 'b', 'LineWidth',2, 'LineStyle','-.')
    else
        rectangle ('Position', [formants_locs(find(formants_locs(:,n),1),n) 0 ...
            (max(formants_locs(:,n))-formants_locs(find(formants_locs(:,n),1),n)) ...
            max(formants_pks(:,n))+4], 'EdgeColor', 'b', 'LineWidth',2, 'LineStyle','-.')
    end
end
hold off

% Finding Formant Ratio

formants_average = zeros(length(formants_pks(1,:)),1);
for i = 1:length(formants_pks(1,:))
    formants_average(i) = mean(formants_pks(:,i))
end


formants_ratio = formants_average./min(formants_average)

%% Male

load("lab4data.mat");
speech = MALE_S;
fs = 12500;
tf = [(1:length(speech))]/fs;
N = length(speech);
M = N - 1; 


% Manual Autocorr 
total = 0;
ACF_speech = zeros(1,length(speech));
for k = 1:M
    for n = 1:N-k+1
        total = total + speech(n).*speech(n+k-1);
    end
    ACF_speech(k) = total;
    total = 0;
end
ACF_speech= ACF_speech./max(ACF_speech);


% Plotting Male Signal Time Domain
figure
plot(tf,speech)
title('Male Speech Time Domain')
xlabel('Time (s)')
ylabel('Amplitude')

% Plotting Autocorr & Finding Pitch Peak Period
NUMBER_PEAKS = 4;
figure
plot(tf,ACF_speech)
[pks, locs] = findpeaks(ACF_speech,tf,'MinPeakDistance',0.003,'NPeaks', NUMBER_PEAKS);
findpeaks(ACF_speech,tf,'MinPeakDistance',0.003,'SortStr','descend','NPeaks', NUMBER_PEAKS);
title('Male Autocorrelation Pitch Detection')
xlabel('Time Lag (s)')
ylabel('Coefficiant of Correlation')

dist = 0;
for i = (1:NUMBER_PEAKS-1)
    dist(i) = locs(i+1)-locs(i);
end

pitch_period = mean(dist)



% Plot of the spectrum 
speech_FFT = abs(fft(speech));
ff_bins = 0:N-1;
ff_hz = ff_bins*fs/N;
figure
plot(ff_hz(1:M/2), speech_FFT(1:M/2));
xlabel('Frequency (Hz)');
ylabel('Magnitude Spectrum');
title('Speech Spectrum of Male Voice');
axis tight; 
Select_FFT = speech_FFT(1:M/2);
% Finding all peaks in signal
[f_pks, f_locs] = findpeaks(Select_FFT,ff_hz(1:M/2),'MinPeakDistance',80,'MinPeakProminence',4);
hold on
plot(f_locs,f_pks,'o')
% Fitting a polynomial line with order = 10 to the minimums of line
p = polyfit(f_locs, -f_pks',10);     % f_pks is negative to find peaks
v = polyval(p, f_locs);
plot(f_locs,v)
% Finding x-axis boundaries of formants by finding peaks of Polynomial Line
[seg_pks, seg_locs] = findpeaks(v,f_locs,'MinPeakDistance',200);
plot (seg_locs,seg_pks,'o') 


% Creating Formant Arrays containing peak and location
formants_pks = 0;
formants_locs = 0;
for n = 1:length(seg_locs)
    for i = 1:length(f_locs)
        if (n == 1) && (f_locs(i) < seg_locs(n))
            formants_pks(i,n) = f_pks(i);
            formants_locs(i,n) = f_locs(i);
        elseif (n == length(seg_locs)) && (f_locs(i) > seg_locs(n))
            formants_pks(i,n+1) = f_pks(i);
            formants_locs(i,n+1) = f_locs(i);
        elseif (f_locs(i) ==  seg_locs(n))
            formants_pks(i,n) = f_pks(i);
            formants_locs(i,n) = f_locs(i);      
        elseif (f_locs(i) <  seg_locs(n)) && (f_locs(i) > seg_locs(n-1))
            formants_pks(i,n) = f_pks(i);
            formants_locs(i,n) = f_locs(i);
        end
    end
end

%Creating Rectangle around each formant 
for n = 1:length(formants_pks(1,:))
    if (n==1)
        rectangle('Position', [0 0 seg_locs(1,n) max(formants_pks(:,n))+4], ...
            'EdgeColor', 'b', 'LineWidth',2, 'LineStyle','-.')
    elseif (n == length(formants_pks(1,:)))
        rectangle ('Position', [formants_locs(find(formants_locs(:,n),1),n)-100 0 ...
            (max(formants_locs(:,n))-formants_locs(find(formants_locs(:,n),1),n)+200) ...
            max(formants_pks(:,n))+4], 'EdgeColor', 'b', 'LineWidth',2, 'LineStyle','-.')
    else
        rectangle ('Position', [formants_locs(find(formants_locs(:,n),1),n) 0 ...
            (max(formants_locs(:,n))-formants_locs(find(formants_locs(:,n),1),n)) ...
            max(formants_pks(:,n))+4], 'EdgeColor', 'b', 'LineWidth',2, 'LineStyle','-.')
    end
end
hold off
    
formants_average = zeros(length(formants_pks(1,:)),1);
for i = 1:length(formants_pks(1,:))
    formants_average(i) = mean(formants_pks(:,i))
end

formants_ratio = formants_average./min(formants_average)

