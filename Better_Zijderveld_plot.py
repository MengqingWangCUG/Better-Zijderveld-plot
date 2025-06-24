import matplotlib
from past.utils import old_div
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ultraplot as upt
from matplotlib.pyplot import MultipleLocator
from numpy import linalg
import matplotlib.ticker as ticker
from adjustText import adjust_text
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 8})
colors = ['#DE3F2A',
    '#F78404',
    '#F9D83F',
    '#6A961F',
    '#15A99E',
    '#0487E2',
    '#804595',
    '#89678B'
]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

def NSzplot(dec, inc, moment, 
            specimen_name=None,
            Iunit='$Am^2$',
            treatmenttext=None,
            ax=None,
            PCA=False,
            PCAdf=None,
            generaldf=None,
            selectlow=None,
            selecthigh=None,
            selectbool=None,
            markersize=4,
            linewidth=1,
            ):
    """
    Create a North-South-Up Zijderveld plot.
    Parameters:
    -----------
    dec : pandas.Series or array-like
        Declination values
    inc : pandas.Series or array-like
        Inclination values  
    moment : pandas.Series or array-like
        Magnetic moment values
    specimen_name : str, optional
        Name of the specimen for plot title
    Iunit : str, optional
        Unit of magnetic moment (default: '$Am^2$')
    treatmenttext : pandas.DataFrame or array-like, optional
        Treatment step labels
    ax : matplotlib.axes, optional
        Axes to plot on
    PCA : bool, optional
        Whether to include PCA analysis (default: False)
    PCAdf : pandas.DataFrame, optional
        PCA results dataframe
    generaldf : pandas.DataFrame, optional
        General data for PCA plotting
    selectlow : float, optional
        Lower bound for PCA selection
    selecthigh : float, optional
        Upper bound for PCA selection
    selectbool : array-like, optional
        Boolean array for PCA point selection
    markersize : int, optional
        Size of markers (default: 4)
    linewidth : int, optional
        Width of lines (default: 1)
    Returns:
    --------
    matplotlib.axes
        The axes object with the plot
    """
    
    if ax is None:
        fig, ax = upt.subplots(figsize=(18/2.54, 18/2.54), ncols=1, nrows=1, share=False)
    if not isinstance(dec, pd.Series):
        dec = pd.Series(dec)
    if not isinstance(inc, pd.Series):
        inc = pd.Series(inc)
    if not isinstance(moment, pd.Series):
        moment = pd.Series(moment)
    VH = dir2cart(pd.concat([90-dec, -inc, moment], axis=1))
    ax.plot(VH[:,0], VH[:,1],
        color='black', linestyle='-', marker='o', markersize=markersize, 
        linewidth=linewidth, markerfacecolor='black', markeredgecolor='black', label='N')
    ax.plot(VH[:,0], VH[:,2],
        color='black', linestyle='-', marker='o', markersize=markersize, 
        linewidth=linewidth, markerfacecolor='white', markeredgecolor='black', label='UP')
    ax.plot(0, 0, alpha=0)
    if PCA and PCAdf is not None:
        try:
            redo_pcaplot(ax, generaldf, VH, PCAdf, selectlow, selecthigh, selectbool, 
                        markersize=markersize, markeredgewidth=None)
        except Exception as e:
            print("Error occurred while performing PCA:", e)
    maxX = np.max(np.maximum(0, VH[:,0]))
    maxY = np.max(np.maximum(0, np.maximum(VH[:,1], VH[:,2])))
    maxall = max(maxX, maxY)
    minX = np.min(np.minimum(0, VH[:,0]))
    minY = np.min(np.minimum(0, np.minimum(VH[:,1], VH[:,2])))
    minall = min(minX, minY)
    tick_step = calculate_nice_tick_step(minall, maxall)
    realrange = [minall-tick_step, maxall+1.5*tick_step]
    xvisrange = [minX-tick_step, maxX+tick_step]
    yvisrange = [minY-tick_step, maxY+tick_step]
    setup_partial_axis(ax, x_full_range=realrange, y_full_range=realrange, 
                      x_visible_range=xvisrange, y_visible_range=yvisrange)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    max_exp = int(np.floor(np.log10(tick_step)))
    title = specimen_name if specimen_name else 'Zijderveld Plot'
    ax.format(title=title, 
              grid=False, xloc='zero', yloc='zero', tickminor=False, ticklabelsize=6)
    if abs(max_exp) >= 2:
        ax.xaxis.set_major_formatter(precision_formatter(ax, 'x', max_exp))
        ax.yaxis.set_major_formatter(precision_formatter(ax, 'y', max_exp))
        
        xlabeltext = r'$\mathrm{E\;(\times 10^{' + str(max_exp) + r'} ' + Iunit.strip('$') + r')}$'
        ylabeltext = r'N ● UP ○ ($\times 10^{' + str(max_exp) + r'} ' + Iunit.strip('$') + r'$)'
    else:
        xlabeltext = r'$\mathrm{E\;(' + Iunit.strip('$') + r')}$'
        ylabeltext = r'N ● UP ○ ($' + Iunit.strip('$') + r'$)'
    ax.text(0, yvisrange[1]+tick_step*0.15, ylabeltext, fontname='DejaVu Sans',
            ha='center', va='bottom', fontsize=6, clip_on=False)
    ax.text(xvisrange[1]+tick_step*0.3, -tick_step*0.15, xlabeltext, fontname='DejaVu Sans',
            va='center', fontsize=6, rotation=270, clip_on=False)
    if treatmenttext is not None:
        steps = []
        if isinstance(treatmenttext, pd.DataFrame):
            treatmtext = np.array(treatmenttext.values).flatten().tolist()
        else:
            treatmtext = list(treatmenttext)
        
        plottreatmtext = treatmtext[::3]
        textVH = VH[::3]
        
        for t in range(len(textVH)):
            x = float(np.array(textVH)[t, 0])
            y = float(np.array(textVH)[t, 1])
            steps.append(ax.text(x, y, plottreatmtext[t], ha='center', va='center', 
                                color='gray', fontsize=6))
        adjust_text(steps, ax=ax)
    
    return ax

def calculate_nice_tick_step(data_min, data_max, density=0.5):
    data_range = max(data_max - data_min, 1e-10) 
    magnitude = 10 ** np.floor(np.log10(data_range))
    normalized_range = data_range / magnitude
    if normalized_range < 1.5:
        step = 0.25 * magnitude / density
    elif normalized_range < 3:
        step = 0.5 * magnitude / density 
    elif normalized_range < 7:
        step = 1.0 * magnitude / density
    else:
        step = 2.5 * magnitude / density
    min_ticks = 4
    if (data_max - data_min) / step < min_ticks:
        step /= 2
        
    return step

def setup_partial_axis(ax, x_full_range=None, y_full_range=None, x_visible_range=None, y_visible_range=None):
    if x_full_range is not None:
        ax.set_xlim(x_full_range)
        if x_visible_range is not None:
            class XPartialLocator(ticker.MaxNLocator):
                def __call__(self):
                    ticks = super().__call__()
                    return [t for t in ticks if x_visible_range[0] <= t <= x_visible_range[1]]
            ax.xaxis.set_major_locator(XPartialLocator())
            ax.spines['bottom'].set_bounds(x_visible_range[0], x_visible_range[1])
    
    if y_full_range is not None:
        ax.set_ylim(y_full_range)
        if y_visible_range is not None:
            class YPartialLocator(ticker.MaxNLocator):
                def __call__(self):
                    ticks = super().__call__()
                    return [t for t in ticks if y_visible_range[0] <= t <= y_visible_range[1]]
            ax.yaxis.set_major_locator(YPartialLocator())
            ax.spines['left'].set_bounds(y_visible_range[0], y_visible_range[1])
    return ax

def precision_formatter(ax, axis='x', exp=0):
    if axis == 'x':
        ticks = ax.get_xticks()
    else:
        ticks = ax.get_yticks()
    scaled_ticks = [t / (10**exp) for t in ticks if abs(t) > 1e-15]
    
    if len(scaled_ticks) > 1:
        intervals = [abs(scaled_ticks[i] - scaled_ticks[i-1]) 
                    for i in range(1, len(scaled_ticks))]
        min_interval = min(intervals)
        decimal_places = max(2, min(4, int(-np.log10(min_interval)) + 1))
    else:
        decimal_places = 2
    
    def formatter(x, pos):
        if abs(x) < 1e-15:
            return '0'
        
        scaled_value = x / (10**exp)
        if abs(scaled_value - round(scaled_value)) < 1e-10:
            return f'{int(round(scaled_value))}'
        formatted = f'{scaled_value:.{decimal_places}f}'

        parts = formatted.split('.')
        if len(parts) > 1:
            decimal_part = parts[1].rstrip('0')  
            if decimal_part:  
                return f'{parts[0]}.{decimal_part}'
            else:  
                return parts[0]
        return formatted
    
    return ticker.FuncFormatter(formatter)

def redo_pcaplot(ax,SDATAI,VH,PCAdf,startstep,endstep,Select_bool,markersize=4,markeredgewidth=0.3):
    # Perform PCA and plot the results
    for u in range(len(PCAdf)):
        
        Spcadf=PCAdf.loc[[u],:]
        PREVH=VH[Select_bool,:]
        starti=SDATAI.loc[SDATAI['treatment']==startstep,:].index[0]
        endi=SDATAI.loc[SDATAI['treatment']==endstep,:].index[0]
        startint=SDATAI.loc[starti,'moment']
        endint=SDATAI.loc[endi,'moment']
        if Spcadf['calculation_type'][u]=='DE-BFL-A':
            HPCADIR=dir2cart([90-Spcadf.loc[u,'specimen_dec'],-Spcadf.loc[u,'specimen_inc'],startint])#HARROWSCALE
            Varrowx=HPCADIR[0]
            Varrowy=HPCADIR[1]
            Harrowx=HPCADIR[0]
            Harrowy=HPCADIR[2]
            Vtextx=0
            Vtexty=0
            Htextx=0
            Htexty=0
        elif Spcadf['calculation_type'][u]=='DE-BFL-O':

            HPCADIR=dir2cart([90-Spcadf.loc[u,'specimen_dec'],-Spcadf.loc[u,'specimen_inc'],startint])#HARROWSCALE
            VPCADIR=dir2cart([90-Spcadf.loc[u,'specimen_dec'],-Spcadf.loc[u,'specimen_inc'],endint])
            halfalllength=(HPCADIR-VPCADIR)/2*1.2
            zero_row = np.zeros((1, PREVH.shape[1]))
            meanpoint=np.mean(np.vstack((PREVH, zero_row)), axis=0)
            arrowpoint=meanpoint+halfalllength
            textpoint=meanpoint-halfalllength
            Varrowx=arrowpoint[0]
            Varrowy=arrowpoint[1]
            Harrowx=arrowpoint[0]
            Harrowy=arrowpoint[2]
            Vtextx=textpoint[0]
            Vtexty=textpoint[1]
            Htextx=textpoint[0]
            Htexty=textpoint[2]
        else:
            HPCADIR=dir2cart([90-Spcadf.loc[u,'specimen_dec'],-Spcadf.loc[u,'specimen_inc'],startint])#HARROWSCALE
            VPCADIR=dir2cart([90-Spcadf.loc[u,'specimen_dec'],-Spcadf.loc[u,'specimen_inc'],endint])
            halfalllength=(HPCADIR-VPCADIR)/2*1.2
            meanpoint=np.mean(PREVH, axis=0)
            arrowpoint=meanpoint+halfalllength
            textpoint=meanpoint-halfalllength
            Varrowx=arrowpoint[0]
            Varrowy=arrowpoint[1]
            Harrowx=arrowpoint[0]
            Harrowy=arrowpoint[2]
            Vtextx=textpoint[0]
            Vtexty=textpoint[1]
            Htextx=textpoint[0]
            Htexty=textpoint[2]
        arrow_style = "simple,head_length=0.6,head_width=0.6,tail_width=0.2"
        ax.plot(PREVH[:,0],PREVH[:,1],'o',markersize=markersize,markerfacecolor='none',markeredgecolor='#E45F2B',markeredgewidth=0.5,)   
        ax.plot(PREVH[:,0],PREVH[:,2],'o',markersize=markersize,markerfacecolor='none',markeredgecolor='#2A88FA',markeredgewidth=0.5,)   

        ax.annotate("",xy=(Harrowx*1.3,Harrowy*1.3),\
                    xytext=(Htextx,Htexty),\
                    arrowprops=dict(arrowstyle=arrow_style,mutation_scale=10, linewidth=0.1,color='#2A88FA',alpha=0.7),
                    annotation_clip=False)
        ax.annotate("",xy=(Varrowx*1.3,Varrowy*1.3),\
                    xytext=(Vtextx,Vtexty),\
                    arrowprops=dict(arrowstyle=arrow_style,mutation_scale=10,linewidth=0.1,color='#E45F2B',alpha=0.7),
                    annotation_clip=False)
def dir2cart(d):
    rad = np.pi / 180.0
    d = np.array(d, dtype=float)
    is_single = len(d.shape) == 1
    if is_single:
        d = d.reshape(1, -1)
    decs = d[:, 0] * rad
    incs = d[:, 1] * rad
    ints = d[:, 2] if d.shape[1] == 3 else np.ones(len(d))
    x = ints * np.cos(decs) * np.cos(incs)
    y = ints * np.sin(decs) * np.cos(incs)
    z = ints * np.sin(incs)
    result = np.column_stack([x, y, z])
    return result[0] if is_single else result


def redo_pca(redodf,generaldf,treatment_type,specimen_name):

    redodf[0]=redodf[0].str.replace('current_', '')
    #Single_redo=redodf
    vals2 = redodf.iloc[:, 2].to_numpy()
    vals3 = redodf.iloc[:, 3].to_numpy()
    redodf.iloc[:, 2] = np.where(vals2 < 1, vals2 * 1000, vals2 - 273)
    redodf.iloc[:, 3] = np.where(vals3 < 1, vals3 * 1000, vals3 - 273)
    #Single_redo = redodf.copy()
    fit_number=len(redodf)
    #PCAmethod=pd.DataFrame(['DE-BFL', 'DE-BFL-A', 'DE-BFL-O','DE-BFP', 'DE-FM'])
    for u in range(fit_number):
        fitname=redodf.iloc[u,5]
        Single_redo=redodf.loc[u,:]
        starti=generaldf.loc[generaldf['treatment']==Single_redo[2],:].index[0]
        endi=generaldf.loc[generaldf['treatment']==Single_redo[3],:].index[0]
        selectlow=Single_redo[2]
        selecthigh=Single_redo[3]
        selectbool=(generaldf['treatment']>=selectlow)&(generaldf['treatment']<=selecthigh)
        PCAmethod=Single_redo[1]
        try:
            testq=generaldf['quality']
        except:
            generaldf['quality']='g'
        try:
            data_g=generaldf[['treatment','dec_g','inc_g','moment','quality']].values.tolist()
            pca_g=pd.DataFrame(domean(data_g,starti,endi,PCAmethod))
            fisher_g=pd.DataFrame(domean(data_g,starti,endi,'DE-FM'))
            pca_g['specimen']=specimen_name[0]
            pca_g['fit_name']=fitname
            fisher_g['specimen']=specimen_name[0]
            fisher_g['fit_name']=fitname
            fisher_g=fisher_g.drop_duplicates(keep='first')
            center_of_mass_col = pca_g.pop('center_of_mass')
            pca_g=pca_g.drop_duplicates(keep='first')
            transposed_center_of_mass = center_of_mass_col.values.reshape(1, -1)
            new_columns = ['center_of_mass1', 'center_of_mass2', 'center_of_mass3']
            df_transposed = pd.DataFrame(transposed_center_of_mass, columns=new_columns)
            pca_g = pd.concat([pca_g, df_transposed], axis=1)
        except:
            pca_g=pd.DataFrame([])
            fisher_g=pd.DataFrame([])
        try:
            data_t=generaldf[['treatment','dec_t','inc_t','moment','quality']].values.tolist()
            pca_t=pd.DataFrame(domean(data_t,starti,endi,PCAmethod))
            fisher_t=pd.DataFrame(domean(data_t,starti,endi,'DE-FM'))
            pca_t['specimen']=specimen_name[0]
            pca_t['fit_name']=fitname
            fisher_t['specimen']=specimen_name[0]
            fisher_t['fit_name']=fitname
            fisher_t=fisher_t.drop_duplicates(keep='first')
            center_of_mass_col = pca_t.pop('center_of_mass')
            pca_t=pca_t.drop_duplicates(keep='first')
            transposed_center_of_mass = center_of_mass_col.values.reshape(1, -1)
            new_columns = ['center_of_mass1', 'center_of_mass2', 'center_of_mass3']
            df_transposed = pd.DataFrame(transposed_center_of_mass, columns=new_columns)
            pca_t = pd.concat([pca_t, df_transposed], axis=1)
        except:
            pca_t=pd.DataFrame([])
            fisher_t=pd.DataFrame([])
        try:
            data_s=generaldf[['treatment','dec_s','inc_s','moment','quality']].values.tolist()
            pca_s=pd.DataFrame(domean(data_s,starti,endi,PCAmethod))
            fisher_s=pd.DataFrame(domean(data_s,starti,endi,'DE-FM'))
            pca_s['specimen']=specimen_name[0]
            pca_s['fit_name']=fitname
            fisher_s['specimen']=specimen_name[0]
            fisher_s['fit_name']=fitname
            fisher_s=fisher_s.drop_duplicates(keep='first')
            center_of_mass_col = pca_s.pop('center_of_mass')
            pca_s=pca_s.drop_duplicates(keep='first')
            transposed_center_of_mass = center_of_mass_col.values.reshape(1, -1)
            new_columns = ['center_of_mass1', 'center_of_mass2', 'center_of_mass3']
            df_transposed = pd.DataFrame(transposed_center_of_mass, columns=new_columns)
            pca_s = pd.concat([pca_s, df_transposed], axis=1)
        except:
            pca_s=pd.DataFrame([])
            fisher_s=pd.DataFrame([])
    return selectbool,selectlow,selecthigh,pca_g,fisher_g,pca_t,fisher_t,pca_s,fisher_s


def domean(data, start, end, calculation_type):
    mpars = {}
    datablock = []
    start0, end0 = start, end
    indata = []
    for rec in data:
        if len(rec) < 6:
            rec.append('g')
        indata.append(rec)
    if indata[start0][5] == 'b':
        print("Can't select 'bad' point as start for PCA")
    flags = [x[5] for x in indata]
    bad_before_start = flags[:start0].count('b')
    bad_in_mean = flags[start0:end0 + 1].count('b')
    start = start0 - bad_before_start
    end = end0 - bad_before_start - bad_in_mean
    datablock = [x for x in indata if x[5] == 'g']
    if indata[start0] != datablock[start]:
        print('problem removing bad data in domean start of datablock shifted:\noriginal: %d\nafter removal: %d' % (
            start0, indata.index(datablock[start])))
    if indata[end0] != datablock[end]:
        print('problem removing bad data in domean end of datablock shifted:\noriginal: %d\nafter removal: %d' % (
            end0, indata.index(datablock[end])))
    mpars["calculation_type"] = calculation_type
    rad = np.pi/180.
    if end > len(datablock) - 1 or end < start:
        end = len(datablock) - 1
    control, data, X, Nrec = [], [], [], float(end - start + 1)
    cm = [0., 0., 0.]
    fdata = []
    for k in range(start, end + 1):
        if calculation_type == 'DE-BFL' or calculation_type == 'DE-BFL-A' or calculation_type == 'DE-BFL-O':  # best-fit line
            data = [datablock[k][1], datablock[k][2], datablock[k][3]]
        else:
            data = [datablock[k][1], datablock[k][2], 1.0]  # unit weight
        fdata.append(data)
        cart = dir2cart(data)
        X.append(cart)
    if calculation_type == 'DE-BFL-O':  # include origin as point
        X.append([0., 0., 0.])
    for cart in X:
        for l in range(3):
            cm[l] += cart[l] / Nrec
    mpars["center_of_mass"] = cm
    if calculation_type != 'DE-BFP':
        mpars["specimen_direction_type"] = 'l'
    if calculation_type == 'DE-BFL' or calculation_type == 'DE-BFL-O':  # not for planes or anchored lines
        for k in range(len(X)):
            for l in range(3):
                X[k][l] = X[k][l] - cm[l]
    else:
        mpars["specimen_direction_type"] = 'p'
    T = np.array(Tmatrix(X))
    t, V = tauV(T)
    if t == []:
        mpars["specimen_direction_type"] = "Error"
        print("Error in calculation")
        return mpars
    v1, v3 = V[0], V[2]
    if t[2] < 0:
        t[2] = 0  # make positive
    if calculation_type == 'DE-BFL-A':
        Dir, R = vector_mean(fdata)
        mpars["specimen_direction_type"] = 'l'
        mpars["specimen_dec"] = Dir[0]
        mpars["specimen_inc"] = Dir[1]
        mpars["specimen_n"] = len(fdata)
        mpars["measurement_step_min"] = indata[start0][0]
        mpars["measurement_step_max"] = indata[end0][0]
        mpars["center_of_mass"] = cm
        s1 = np.sqrt(t[0])
        MAD = np.arctan(np.sqrt(t[1] + t[2]) / s1) / rad
        if np.iscomplexobj(MAD):
            MAD = MAD.real
        # I think this is how it is done - i never anchor the "PCA" - check
        mpars["specimen_mad"] = MAD
        return mpars
    if calculation_type != 'DE-BFP':
        rec = [datablock[start][1], datablock[start][2], datablock[start][3]]
        P1 = dir2cart(rec)
        rec = [datablock[end][1], datablock[end][2], datablock[end][3]]
        P2 = dir2cart(rec)

        for k in range(3):
            control.append(P1[k] - P2[k])
        dot = 0
        for k in range(3):
            dot += v1[k] * control[k]
        if dot < -1:
            dot = -1
        if dot > 1:
            dot = 1
        if np.arccos(dot) > old_div(np.pi, 2.):
            for k in range(3):
                v1[k] = -v1[k]
        s1 = np.sqrt(t[0])
        Dir = cart2dir(v1)
        MAD = old_div(np.arctan(old_div(np.sqrt(t[1] + t[2]), s1)), rad)
        if np.iscomplexobj(MAD):
            MAD = MAD.real
    if calculation_type == "DE-BFP":
        Dir = cart2dir(v3)
        MAD = old_div(
            np.arctan(np.sqrt(old_div(t[2], t[1]) + old_div(t[2], t[0]))), rad)
        if np.iscomplexobj(MAD):
            MAD = MAD.real
    CMdir = cart2dir(cm)
    Dirp = [Dir[0], Dir[1], 1.]
    dang = angle(CMdir, Dirp)
    mpars["specimen_dec"] = Dir[0]
    mpars["specimen_inc"] = Dir[1]
    mpars["specimen_mad"] = MAD
    # mpars["specimen_n"]=int(Nrec)
    mpars["specimen_n"] = len(X)
    mpars["specimen_dang"] = dang[0]
    mpars["measurement_step_min"] = indata[start0][0]
    mpars["measurement_step_max"] = indata[end0][0]
    return mpars


def cart2dir(cart):
    cart = np.array(cart)
    rad = np.pi/180.  
    if len(cart.shape) > 1:
        Xs, Ys, Zs = cart[:, 0], cart[:, 1], cart[:, 2]
    else:  # single vector
        Xs, Ys, Zs = cart[0], cart[1], cart[2]
    if np.iscomplexobj(Xs):
        Xs = Xs.real
    if np.iscomplexobj(Ys):
        Ys = Ys.real
    if np.iscomplexobj(Zs):
        Zs = Zs.real
    Rs = np.sqrt(Xs**2 + Ys**2 + Zs**2) 
    Decs = (np.arctan2(Ys, Xs) / rad) % 360.
    try:
        Incs = np.arcsin(Zs / Rs) / rad
    except:
        print('trouble in cart2dir') 
        return np.zeros(3)
    direction_array = np.array([Decs, Incs, Rs]).transpose() 
    return direction_array  
def vector_mean(data):
    Xbar = np.zeros((3))
    X = dir2cart(data).transpose()
    for i in range(3):
        Xbar[i] = X[i].sum()
    R = np.sqrt(Xbar[0]**2+Xbar[1]**2+Xbar[2]**2)
    Xbar = Xbar/R
    dir = cart2dir(Xbar)
    return dir, R

def angle(D1, D2):
    D1 = np.array(D1)
    if len(D1.shape) > 1:
        D1 = D1[:, 0:2]  # strip off intensity
    else:
        D1 = D1[:2]
    D2 = np.array(D2)
    if len(D2.shape) > 1:
        D2 = D2[:, 0:2]  # strip off intensity
    else:
        D2 = D2[:2]
    X1 = dir2cart(D1)  # convert to cartesian from polar
    X2 = dir2cart(D2)
    angles = []  # set up a list for angles
    for k in range(X1.shape[0]):  # single vector
        angle = np.arccos(np.dot(X1[k], X2[k])) * \
            180. / np.pi  # take the dot product
        angle = angle % 360.
        angles.append(angle)
    return np.array(angles)

def tauV(T):
    t, V, tr = [], [], 0.
    ind1, ind2, ind3 = 0, 1, 2
    evalues, evectmps = linalg.eig(T)
    evectors = np.transpose(evectmps)
    for tau in evalues:
        tr += tau
    if tr != 0:
        for i in range(3):
            evalues[i] = evalues[i] / tr
    else:
        return t, V
    t1, t2, t3 = 0., 0., 1.
    for k in range(3):
        if evalues[k] > t1:
            t1, ind1 = evalues[k], k
        if evalues[k] < t3:
            t3, ind3 = evalues[k], k
    for k in range(3):
        if evalues[k] != t1 and evalues[k] != t3:
            t2, ind2 = evalues[k], k
    V.append(evectors[ind1])
    V.append(evectors[ind2])
    V.append(evectors[ind3])
    t.append(t1)
    t.append(t2)
    t.append(t3)
    return t, V


def Tmatrix(X):
    T = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    for row in X:
        for k in range(3):
            for l in range(3):
                T[k][l] += row[k] * row[l]
    return T
