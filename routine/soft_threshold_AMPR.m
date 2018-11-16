function [out]=mysoft_th(A,B,lambda)
Llam=length(lambda);
LB=length(B);
out=zeros(LB,Llam);

for ilam=1:Llam
out(:,ilam)=(B-sign(B)*lambda(ilam)).*(abs(B) > lambda(ilam))./A;
end

end