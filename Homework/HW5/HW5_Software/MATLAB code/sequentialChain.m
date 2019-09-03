function [pos, w]=sequentialChain(N,option)
% positions of the initial two molecules

pos(1,:)=[0 0];
w = 1;
i = 2;

while (pos(end,1)~=N)||(pos(end,2)~=N)
    if option == 1
        cand_pos=[pos(i-1,1)+[0;1] pos(i-1,2)+[1;0]];   % 2 monotonic moves
    else
        cand_pos=[pos(i-1,1)+[0;0;-1;1] pos(i-1,2)+[1;-1;0;0]]; % 4 moves
    end
    
    % look for the admissible neighboring positions
    mark=zeros(size(cand_pos,1),1);         % mark the occupied neighboring positions
    for j=1:size(cand_pos,1)
        for m=1:size(pos,1)
            if cand_pos(j,:)==pos(m,:),
                mark(j)=1;
                break;
            end
        end
        if (max(cand_pos(j,:))>N)||(min(cand_pos(j,:))<0)
            mark(j) = 1;
        end
    end
    cand_pos(find(mark==1),:)=[];
    
    if length(cand_pos)==0,
        pos=[];
        w = 0;
        return;
    else
        ind=unidrnd(size(cand_pos,1));
        pos(i,:)=cand_pos(ind,:);
        w = w*size(cand_pos,1);
    end
    i = i+1;
end

