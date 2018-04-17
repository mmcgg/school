function drawRobot(x,y,theta,mark,color)


hold on

r = 0.5;
th = 0:pi/50:2*pi;
xunit = r*cos(th)+x;
yunit = r*sin(th)+y;
plot(xunit,yunit,color)
line([x, r*cos(theta)+x], [ y, r*sin(theta)+y])
plot(x,y,'color', color)


for i =1:size(mark,1)
    plot(mark(i,1), mark(i,2),strcat(color,'x'))
end
hold off
