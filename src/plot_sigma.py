#!/usr/bin/python3

from numpy import *
from matplotlib.pylab import *
from slope_marker import slope_marker

rcParams["figure.figsize"] = [8,7]
rcParams["figure.dpi"] = 200

def plot_order_line(ax, h_list):
    n=len(h_list)
    x=np.array([h_list[n-1], h_list[0]])
    y=x
    ax.plot(x,y,'-',color='gray',lw=4, alpha=0.6)
    y=x**2
    ax.plot(x,y,'-',color='gray',lw=5, alpha=0.6)
    legends = ['Order1', 'Order 2']
    return legends

def plot_triangle(ax, h_list, y_position, slope=1):
    n=len(h_list)
    x=np.array([h_list[n-1], h_list[0]])
    y0=y_position
    y1=y0+slope*(x[1]-x[0])
    y=[y0, y1]
    print ("x, y=",x, y)
    vertices = [ [x[0], y[0]], [x[1],y[0]], [x[1],y[1]] ]
    from matplotlib.patches import Polygon
    triang = Polygon(vertices)
    ax.add_patch( triang )

error_files = ["erroresL2SIP_sigma","erroresSIP_sigma"]
error_names = ["L2 norm", "SIP norm"]

line_styles=["-","-"]
line_markers=["^","s"]

errors={k:None for k in error_files}

for e in error_files:
    filename = e + ".txt"
    with open(filename, 'r') as f:
        errors[e] = array([float(x) for x in f.readlines()])

n = len(list(errors.values())[0])
h0 = 1
x = array([0.01, 0.5, 1, 2, 3, 4, 5, 6 ])

fig = figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_yscale('log')

for e,l,s,m in zip(error_files,error_names,line_styles,line_markers):
    y = errors[e]
    ax.plot(x,y,linestyle=s,linewidth=4,label=l,marker=m,markersize=8)
    print(e, "x=", x, "y=", y)

xlabel("Sigma")
ylabel("Error")
legend()

print(errors.keys())
print(errors.values())
ymin = min([min(errors[k]) for k in errors.keys()])
ymax = max([max(errors[k]) for k in errors.keys()])
# ymin, ymax = min(min(errors.values())), max(max(errors.values()))
print(ymin, ymax)

# plot_order_line(ax, x)
fig.savefig('../../../../../Graficas/Difusion/errores_difusion_sigma.png')

show()
