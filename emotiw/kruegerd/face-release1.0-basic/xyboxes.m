function [xs,ys]=xyboxes(im, boxes, posemap)

disp('boxes');
disp(boxes);

xs = [];
ys = [];

for b = boxes,
    partsize = b.xy(1,3)-b.xy(1,1)+1;
    tx = (min(b.xy(:,1)) + max(b.xy(:,3)))/2;
    ty = min(b.xy(:,2)) - partsize/2;
    text(tx,ty, num2str(posemap(b.c)),'fontsize',18,'color','c');
    for i = size(b.xy,1):-1:1;
        x1 = b.xy(i,1);
        y1 = b.xy(i,2);
        x2 = b.xy(i,3);
        y2 = b.xy(i,4);
        xs(i)=(x1+x2)/2; ys(i)=(y1+y2)/2;
    end
end
