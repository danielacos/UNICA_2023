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

error_files = ["erroresL2CF","erroresCF","erroresL2UPW","erroresUPW"]
error_names = ["uCF, norm L2", "uCF, norm CF", "uUPW, norm L2", "uUPW, norm UPW"]

line_styles=["-","-",":",":"]
line_markers=["*","o","^","s"]

errors={k:None for k in error_files}

for e in error_files:
    filename = e + ".txt"
    with open(filename, 'r') as f:
        errors[e] = array([float(x) for x in f.readlines()])

n = len(list(errors.values())[0])
h0 = pi/8
x = array([ h0/(2**n) for n in range(n) ])

fig = figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xscale('log')
ax.set_yscale('log')

for e,l,s,m in zip(error_files,error_names,line_styles,line_markers):
    y = errors[e]
    ax.plot(x,y,linestyle=s,linewidth=4,label=l,marker=m,markersize=8)
    print(e, "x=", x, "y=", y)

xlabel("Mesh size")
ylabel("Error")
legend()

# Draw (triangle) slope marker
# plot_triangle(ax, x, min(y), slope=5)
marker_size = 0.2
alfax = 0.55
xmin, xmax = min(x), max(x)
x_marker = alfax*xmin + (1-alfax)*xmax

print(errors.keys())
print(errors.values())
ymin = min([min(errors[k]) for k in errors.keys()])
ymax = max([max(errors[k]) for k in errors.keys()])
# ymin, ymax = min(min(errors.values())), max(max(errors.values()))
print(ymin, ymax)

alfay = 0.998
y_marker = alfay*ymin+(1-alfay)*ymax
print(y_marker)
slope_marker((x_marker, y_marker), slope=(1, 1), size_frac=marker_size, pad_frac=0.07)

alfay = 0.9999
y_marker = alfay*ymin+(1-alfay)*ymax
print(y_marker)
slope_marker((x_marker, y_marker), slope=(2, 1), size_frac=marker_size, pad_frac=0.07)

# plot_order_line(ax, x)
fig.savefig('../../../../../Graficas/Conveccion_Reaccion/errores_conveccion_reaccion_P1dc.png')

show()
