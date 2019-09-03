%% Problem 2 a
clear l w
n = 0;
while n<1e3
[pos, ws]=sequentialChain(10,1);
if ws>0
n = n+1;
l(n) = size(pos,1);
w(n) = ws;
end
end
fprintf('Average total number of chains:\n');
mean(w)


%% Problem 2b
clear l w
n = 0;
while n<1e3
[pos, ws]=sequentialChain(10,2);
if ws>0
n = n+1;
l(n) = size(pos,1);
w(n) = ws;
end
end
fprintf('Average total number of chains:\n');
mean(w)
fprintf('Average length of  chains:\n');
sum(l.*w/sum(w))