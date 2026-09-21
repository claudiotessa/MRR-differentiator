close all
clear all 

%k =0.005; %tau_c = 20 ns
%k =0.05; %tau_c = 2 ns
k =0.5; %tau_c = 0.2 ns

%0.005; % .05;  % Example value for k
A=1e10;  %time scaling parameter  

N=1e4;
tspan = [-100e-9 100e-9];
t=linspace(min(tspan),max(tspan),N);
dt=t(2)-t(1);
%in = (A*t).^3.*exp(-((.1*A*t)).^2).*cos(.3*(A*t));
in = exp(-((.1*A*t)).^12); %[2ns]
%in = exp(-((1*A*t)).^12); %[0.2ns]
%in = exp(-((10*A*t)).^12); %[0.02ns]

% Numerical integration 
y=cumsum(in)*dt*A;

t_min=-2;t_max=20;

% Plot the result
figure(1);
subplot(311);hold on;grid on;box on;plot(t*1e9, in./max(in),'k','LineWidth',2)
xlabel('Time [ns]')
ylabel('Input signal y(t)')
xlim([t_min t_max])
set(gca,'fontsize',12)
subplot(312);hold on;grid on;box on;plot(t*1e9, y./max(y),'k','LineWidth',2)
xlabel('Time [ns]')
ylabel('Integral \int y(t)')
%title('Solution of dy/dt + ky = x(t)')
xlim([t_min t_max])
set(gca,'fontsize',12)
subplot(313);hold on;grid on;box on;plot(t*1e9, abs(y./max(y)).^2,'k','LineWidth',2)
xlabel('Time [ns]')
ylabel('| \int y(t) |^2')
%title('Solution of dy/dt + ky = x(t)')
xlim([t_min t_max])
set(gca,'fontsize',12)

% Implementing the integrator with a microring resonator
k_ring=k*A;
tau_c=1/k_ring;  %cavity life time of the RR
R=10e-6;  %radius of the MRR
L_ring=2*pi*R;
c=3e8;
neff=2.4;  %effective index of the MRR waveguide
tau=L_ring/(c/neff); %round trip time 
tau_n=tau_c/tau;
r=sqrt(tau_n/(1+tau_n)); % copling coefficient of the directional coupler of the MRR


time=t;
dt=time(2)-time(1);
in_ring = in;
IN_ring=fftshift(fft(in_ring));

Df=linspace(-1/(2*dt),1/(2*dt),N);
beta=2*pi*Df/c*neff;
H_drop=1/k*(1-r^2)./(1-r^2*exp(-j*beta*L_ring)).*exp(-j*beta*L_ring/2); %frequency domain description of the MRR
H_int=1/tau_c./(j*2*pi*Df); %frequency domain descritpion of the integrator

Out_ring=IN_ring.*H_drop;
Out_int=IN_ring.*H_int;

out_ring=ifft(fftshift(Out_ring));
out_int=ifft(fftshift(Out_int));

figure(2);hold on; grid on, box on
plot(Df/1e9,10*log10(abs(H_drop./max(abs(H_drop))).^2),'r','LineWidth',2)
plot(Df/1e9,10*log10(abs(H_int./max(abs(H_int))).^2),'b','LineWidth',2)
plot(Df/1e9,10*log10(abs(IN_ring./max(abs(IN_ring))).^2),'k','LineWidth',2)
set(gca,'fontsize',12)
ylim([-30 0])
xlim([-5 5])
xlabel('Frequency [GHz]')
ylabel('Spectrum [dB]')

% Plot the result
figure(1);
subplot(312);hold on;grid on; box on;plot(time*1e9, real(out_ring)./max(real(out_ring)),'r','LineWidth',2)
set(gca,'fontsize',12)
subplot(313);hold on;grid on; box on;plot(time*1e9, (abs(out_ring)./max(abs(out_ring))).^2,'r','LineWidth',2)
set(gca,'fontsize',12)

