function drawRobot(x,y,theta,mark, measureRange, measurePhi,t)

savex = [];
savey = [];
xplot = 2.5;
%color = [1,0,0;0 1 0; 0 0 1; 0 0 0; 0 1 1; 1 0 1];
for i = 1:length(t)
    if mod(t(i),xplot) == 0
        savex = [savex;x(i)];
        savey = [savey;y(i)];
    end
end
count = 0;
for i = 1:length(t)
clf;
% axis equal;
axis([-15,15,-15,10])
    hold on
r = 0.5;
th = 0:pi/50:2*pi;
xunit = r*cos(th)+x(i);
yunit = r*sin(th)+y(i);
% plot(x(i),y(i),'x')
plot(xunit,yunit,'b')
line([x(i), r*cos(theta(i))+x(i)], [ y(i), r*sin(theta(i))+y(i)])
plot(x(1:i),y(1:i),'color', [0 .6 0])
if mod(t(i),xplot) == 0
    count = count + 1;
end
    plot(savex(1:count), savey(1:count),'xk')
for j = 1: size(measureRange,2)
R = [rotz(theta(i)), [0; 0; 0]; [0, 0, 0,1]] ;
Trans = [eye(3), [x(i);y(i);0];[0,0,0,1]];
%phiPlot = measurePhi + theta;
xy = [measureRange(i,j)*cos(measurePhi(i,j)); measureRange(i,j)*sin(measurePhi(i,j)); 0; 1];
% xy = [measureRange(i,j)*cos(phiPlot(i,j)); measureRange(i,j)*sin(phiPlot(i,j)); 0; 1];
xybuff = R*xy;
xyNew = Trans*xybuff;
%line([x(i), xyNew(1)],[y(i), xyNew(2)], 'linestyle', '--', 'color', color(j,:))
line([x(i), xyNew(1)],[y(i), xyNew(2)], 'linestyle', '--')
% line([x(i), xy(1)+x(i)],[y(i), xy(2)+y(2)])
end


for i =1:size(mark,1)
    %plot(mark(i,1), mark(i,2),'x', 'color', color(i,:))
    plot(mark(i,1), mark(i,2),'x')
end
pause(0.0001)
end
hold off
