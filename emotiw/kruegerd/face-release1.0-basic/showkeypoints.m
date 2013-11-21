function [xs,ys]=showkeypoints(im, boxes, posemap)

% showboxes(im, boxes)
% Draw boxes on top of image.

imagesc(im);
hold on;
axis image;
axis off;

for b = boxes,
    for i = size(b.xy,1):-1:1;
        x1 = b.xy(i,1);
        y1 = b.xy(i,2);
        x2 = b.xy(i,3);
        y2 = b.xy(i,4);      
        plot((x1+x2)/2,(y1+y2)/2,'r.','markersize',15);
        xs(i)=(x1+x2)/2; ys(i)=(y1+y2)/2;
    end
end

drawnow;
