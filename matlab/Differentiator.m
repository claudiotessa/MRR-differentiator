close all
clear all 

N=1e5;                                % sample number
tspan = [-10e-9 10e-9];               % span of time
t=linspace(min(tspan),max(tspan),N);  % time istants


%Banda del microring
B_MRR=10e9;%

A=1e10; %time scaling parameter  
in = exp(-((.1*A*t)).^12); Mx=4 %[2ns] 
%in = exp(-((1*A*t)).^12); Mx=20 %[0.2ns]
%in = exp(-((10*A*t)).^12); Mx=200  %[0.02ns]
%in = (A*t).^3.*exp(-((.1*A*t)).^2).*cos(.3*(A*t))+(A*t).^3.*exp(-((.1*A*t)).^2).*cos(.05*(2*A*t)+pi/2) ;Mx=2;

x_min=min(tspan)*1e9/Mx;
x_max=max(tspan)*1e9/Mx;

% Numerical  differentiation
y=gradient(in,A*t);

% Plot the result
figure(1);
subplot(311);hold on;grid on;box on;plot(t*1e9, in./max(in),'k','LineWidth',2)
xlabel('Time [ns]')
ylabel('Input signal y(t)')
xlim([x_min x_max])
set(gca,'fontsize',12)
legend('input')
subplot(312);hold on;grid on;box on;plot(t*1e9, y./max(y),'k','LineWidth',2)
xlabel('Time [ns]')
ylabel('Derivative y''(t)')
xlim([x_min x_max])
set(gca,'fontsize',12)
subplot(313);hold on;grid on;box on;plot(t*1e9, abs(y./max(y)).^2,'k','LineWidth',2)
xlabel('Time [ns]')
ylabel('| y''(t) |^2')
xlim([x_min x_max])
set(gca,'fontsize',12)

% Implementing the differentiator with a microring resonator
%k_ring=k*A;

tau_c=1/(pi*B_MRR);  %cavity life time of the RR
R=100e-6;   %radius of the MRR
L_ring=2*pi*R;
c=3e8;
neff=2.4;   %effective index of the MRR waveguide
tau=L_ring/(c/neff);  %round trip time 
tau_n=tau_c/tau;
r=sqrt(tau_n/(1+tau_n));  % copling coefficient of the directional coupler of the MRR 

time=t;
dt=time(2)-time(1);
in_ring = in;
IN_ring=fftshift(fft(in_ring));

Df=linspace(-1/(2*dt),1/(2*dt),N);
beta=2*pi*Df/c*neff;
phi=0;%pi/100; %round trip phase detuning
gamma=r;%r*0.98;  %critical coupling 
H_through=(r-gamma*exp(-j*beta*L_ring+j*phi))./(1-r*gamma*exp(-j*beta*L_ring+j*phi)); %frequency domain description of the MRR
H_diff=j*r./(1-r^2)*tau*2*pi*Df.*exp(-j*tau*2*pi*Df/2); %frequency domain descritpion of the differentiator

Out_ring=IN_ring.*H_through;
Out_diff=IN_ring.*H_diff;

out_ring=ifft(fftshift(Out_ring));
out_diff=ifft(fftshift(Out_diff));

figure;hold on;box on;grid on
plot(Df/1e9,10*log10(abs(H_through./max(abs(H_through))).^2),'r','LineWidth',2)
plot(Df/1e9,10*log10(abs(H_diff./max(1)).^2),'b','LineWidth',2)
plot(Df/1e9,10*log10(abs(IN_ring./max(abs(IN_ring))).^2),'k','LineWidth',2)
legend('MRR output','ideal derivative', 'input signal')
set(gca,'fontsize',12)
ylim([-30 0])
xlim([-10 10])
xlabel('Frequency [GHz]')
ylabel('Spectrum [dB]')

figure(1);
subplot(312);hold on;grid on; box on;plot(time*1e9, real(out_ring)./max(real(out_ring)),'r','LineWidth',2)
subplot(312);hold on;grid on; box on;plot(time*1e9, imag(out_ring)./max(real(out_ring)),'r--','LineWidth',2)
set(gca,'fontsize',12)
legend('ideal derivative','MRR output (real)','MRR output (imag)')
subplot(313);hold on;grid on; box on;plot(time*1e9, (abs(out_ring)./max(abs(out_ring))).^2,'r','LineWidth',2)
set(gca,'fontsize',12)
legend('ideal derivative (power)','MRR output (power)')

