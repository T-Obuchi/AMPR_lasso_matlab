function [out]=soft_threshold(A,B,lambda)
Llam=length(lambda);
sB=size(B);
for ilam=1:Llam
    out{ilam}=zeros(sB);
end

for ilam=1:Llam
out{ilam}=(B-sign(B)*lambda(ilam)).*(abs(B) > lambda(ilam))./A;
end

end