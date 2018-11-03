
"""
How to use it:

import numpy as np
import time
import datetime
import pandas as pd
import sys,os
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/Users/rtahmasbi/Documents/RASOOL/Main_func')

from DescriptiveStatistics import DescriptiveStatistics
del sys.modules['DescriptiveStatistics']
from DescriptiveStatistics import DescriptiveStatistics

DF = pd.read_csv('df_train_hist_new_all.csv',engine='python')

ds = DescriptiveStatistics(outlier_thr = 3, dpi=200, format='pdf')
ds.main(DF,prefix='stat')

from DescriptiveStatistics import DescriptiveStatistics
del sys.modules['DescriptiveStatistics']
from DescriptiveStatistics import DescriptiveStatistics

DF = pd.read_csv('data.csv',engine='python')

ds = DescriptiveStatistics(outlier_thr = 3, dpi=200, format='png')
ds.main(DF,prefix='sn')



# or
df_ds = pd.DataFrame()
for tag in DF.columns:
    x   = DF[tag].copy()
    print(tag)
    df_ds_temp = ds.plot(x,tag,prefix='stat')
    df_ds = df_ds.append(df_ds_temp, ignore_index=True)

df_ds.to_csv('summary_All.csv', sep=',' ,index=False)


"""

#ver 1.0.3
#-
import os
import datetime
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties



# input is pd
class DescriptiveStatistics:
    def __init__(self, format='pdf', dpi=300, outlier_thr = 3.0, outlier_thr_gradient =3.0,  figsize=(15,10), outlier_thr_Tukey=1.5, modified_z_thr=3.5):
        self.dpi = dpi
        self.format = format
        self.outlier_thr = outlier_thr
        self.figsize = figsize
        self.outlier_thr_gradient = outlier_thr_gradient
        self.outlier_thr_Tukey = outlier_thr_Tukey
        self.outlier_thr_conservative = 2.5
        self.modified_z_thr = modified_z_thr
    #
    def plot(self, x=None, var_name='var_name', prefix='stat_model3'):
        if (x.dtype=='O'): # object
            print('----> Error: variable [%s] is Object' % var_name)
            return
        if not x.ndim==1: # dim
            print('----> Error: Dim [%s] is not 1' % var_name)
            return
        #
        sample_size = x.shape[0]
        num_na = sum(np.isnan(x))
        # remove NaN
        x_orig = x.copy()
        x = x[~np.isnan(x)]
        #
        if x.shape[0]<1:
            print('----> Error: variable [%s] is empty' % var_name)
            return
        data_lim_bot    = np.mean(x) - np.std(x) * self.outlier_thr
        data_lim_top    = np.mean(x) + np.std(x) * self.outlier_thr
        data_lim2p5_bot = np.mean(x) - np.std(x) * self.outlier_thr_conservative
        data_lim2p5_top = np.mean(x) + np.std(x) * self.outlier_thr_conservative
        data_min        = np.min(x)
        data_max        = np.max(x)
        if data_min==data_max:
            print('----> Error: variable [%s] is constant' % var_name)
            return
        data_percentile25 = np.percentile(x,25)
        data_percentile75 = np.percentile(x,75)
        IQR = data_percentile75 - data_percentile25
        outliers_Tukey_top = data_percentile75 + 1.5 * IQR
        outliers_Tukey_bot = data_percentile25 - 1.5 * IQR
        # modified Z-score
        x_tilde = np.median(x)
        MAD = np.median(np.abs(x - x_tilde)) #Median absolute deviation
        #Mi = 0.6745*(x-np.median(x))/MAD
        mod_z_top = self.modified_z_thr*MAD/0.6745 + np.median(x)
        mod_z_bot = -self.modified_z_thr*MAD/0.6745 + np.median(x)
        #
        plt.figure(figsize=self.figsize)
        plt.tight_layout()
        plt.suptitle(var_name)
        ##
        plt.subplot(2, 3, 1)
        font = FontProperties().copy()
        font.set_family('DejaVu Sans Mono')
        font.set_size(8)
        text_d = .08
        text_st = .9
        plt.text(0,text_st - 0*text_d,'sample size       = %d' % sample_size, fontproperties=font)
        plt.text(0,text_st - 1*text_d,'num na            = %d' % num_na, fontproperties=font)
        plt.text(0,text_st - 2*text_d,'mean              = %f' % np.mean(x), fontproperties=font)
        plt.text(0,text_st - 3*text_d,'median            = %f' % np.median(x), fontproperties=font)
        plt.text(0,text_st - 4*text_d,'min               = %f' % data_min, fontproperties=font, color='red')
        plt.text(0,text_st - 5*text_d,'max               = %f' % data_max, fontproperties=font, color='red')
        plt.text(0,text_st - 6*text_d,'std               = %f' % np.std(x), fontproperties=font)
        plt.text(0,text_st - 7*text_d,'var               = %f' % np.var(x), fontproperties=font)
        plt.text(0,text_st - 8*text_d,'mean-%.1f*std(x)   = %f' % (self.outlier_thr,data_lim_bot), fontproperties=font)
        plt.text(0,text_st - 9*text_d,'mean+%.1f*std(x)   = %f' % (self.outlier_thr,data_lim_top), fontproperties=font)
        plt.text(0,text_st -10*text_d,'Q25               = %f' % data_percentile25, fontproperties=font, color='red')
        plt.text(0,text_st -11*text_d,'Q75               = %f' % data_percentile75, fontproperties=font, color='red')
        plt.axis('off')
        plt.ylim(0,1)
        #
        # hist
        plt.subplot(2, 3, 2)
        plt.hist(x,25)
        plt.title('Histogram')
        #
        # boxplot
        plt.subplot(2, 3, 3)
        plt.title('Boxplot')
        plt.boxplot(x)
        #
        # plot time
        ax = plt.subplot(2, 3, 4)
        plt.axhline(y=data_lim2p5_top, color='r', linewidth=.8, linestyle='--',label='2.5$\sigma$')
        plt.axhline(y=data_lim2p5_bot, color='r', linewidth=.8, linestyle='--')
        plt.axhline(y=mod_z_top, color='b', linewidth=.8, linestyle='--',label='z_score=3.5')
        plt.axhline(y=mod_z_bot, color='b', linewidth=.8, linestyle='--')
        plt.axhline(y=data_lim_bot,color='g', linewidth=.8,label='3$\sigma$')
        plt.axhline(y=data_lim_top,color='g', linewidth=.8)
        plt.axhline(y=outliers_Tukey_top,color='black', linewidth=.8,label='1.5*IQR')
        plt.axhline(y=outliers_Tukey_bot,color='black', linewidth=.8)
        plt.title('Data (red=%.1f$\sigma$, green=%.1f$\sigma$,\n black=%.1f*IQR, blue=z_score=%.1f)' % (self.outlier_thr_conservative, self.outlier_thr,self.outlier_thr_Tukey,self.modified_z_thr),fontdict={'fontsize':8})
        box = ax.get_position()
        ax.set_position([box.x0,box.y0+box.height*0,
                        box.width, box.height*1])
        fontP = FontProperties()
        fontP.set_size('x-small')
        ax.legend(loc='upper center', bbox_to_anchor=(.5,-.15),fancybox=True,ncol=4,prop=fontP)
        plt.xlabel('Time')
        plt.plot(x_orig, linewidth=.5)
        y_lim_time = plt.ylim()
        #
        # plot derivative
        if not x.dtype=='bool':
            x_der = np.gradient(x_orig)
            x_der_bot = np.mean(x_der) - np.std(x_der) * self.outlier_thr_gradient
            x_der_top = np.mean(x_der) + np.std(x_der) * self.outlier_thr_gradient
            # Modded z_score addition for derivatives
            #MAD = np.median(np.abs(x_der - np.median(x_der))) #Median absolute deviation
            #Mi = 0.6745*(x_der-np.median(x_der))/MAD
            #x_der_mod_z_top = self.modified_z_thr*MAD/0.6745 + np.median(x_der)
            #x_der_mod_z_bot = -self.modified_z_thr*MAD/0.6745 + np.median(x_der)
            # Modded z end
            plt.subplot(2, 3, 5)
            plt.axhline(y=x_der_bot, color='g', linewidth=.8)
            plt.axhline(y=x_der_top, color='g', linewidth=.8)
            plt.axhline(y=np.mean(x_der) - np.std(x_der) * self.outlier_thr_conservative, color='r', linewidth=.8, linestyle='--')
            plt.axhline(y=np.mean(x_der) + np.std(x_der) * self.outlier_thr_conservative ,color='r', linewidth=.8, linestyle='--')
            #plt.axhline(y=x_der_mod_z_top, color='b', linewidth=.8, linestyle='--')
            #plt.axhline(y=x_der_mod_z_bot, color='b', linewidth=.8, linestyle='--')
            plt.title('Gradient (red=%.1f$\sigma$, green=%.1f$\sigma$, blue=z_score = %.1f))' % (self.outlier_thr_conservative,self.outlier_thr_gradient,self.modified_z_thr),fontdict={'fontsize':8})
            plt.xlabel('Time')
            plt.plot(x_der, linewidth=.5)
            #
            #### plot final
            idx_bad1 = (x_orig>=data_lim_top)
            idx_bad2 = (x_orig<=data_lim_bot)
            idx_bad3 = (x_der>=x_der_top)
            idx_bad4 = (x_der<=x_der_bot)
            idx_bad = np.logical_or(np.logical_or(idx_bad1, idx_bad2), np.logical_or(idx_bad3, idx_bad4))
            #
            x_final = x.copy()
            x_final[idx_bad] = np.nan
            #x_final = x_final[~np.isnan(x_final)]
            plt.subplot(2, 3, 6)
            plt.xlabel('Time')
            plt.title('Final (based on %.1f$\sigma$ and Gradient)' % (self.outlier_thr))
            #plt.ylim(y_lim_time)
            plt.plot(x,'r.', markersize=1.2)
            plt.plot(x_final,'b.', markersize=1.2)
            ##################################
            # save info
            d = {'var_name': var_name,
                'data_min': data_min, 'data_lim_bot':data_lim_bot, 'data_lim2p5_bot':data_lim2p5_bot, 'outliers_Tukey_bot':outliers_Tukey_bot,
                'data_max': data_max, 'data_lim_top':data_lim_top, 'data_lim2p5_top':data_lim2p5_top, 'outliers_Tukey_top':outliers_Tukey_top,
                'outlier_thr_gradient':self.outlier_thr_gradient, 'modified_z_score_bot':mod_z_bot, 'modified_z_score_top':mod_z_top}
            df = pd.DataFrame(data=d,index=[0], columns=d.keys())
            #fname_info = ('summary_%s.csv' % (var_name))
            #df.to_csv(fname_info, sep=',' ,index=False)
        #
        ##################################
        var_name = var_name.replace('/','_')
        fname = ('%s_%s.%s' % (prefix,var_name,self.format))
        #plt.show()
        plt.savefig(fname, dpi=self.dpi, format=self.format)
        plt.close()
        return(df) #returns descriptive stats table and clean data

    def main(self,DF, prefix='stat_model3'):
        df_ds = pd.DataFrame()
        for tag in DF.columns:
            x   = DF[tag].copy()
            print(tag)
            df_ds_temp = self.plot(x,tag, prefix)
            df_ds = df_ds.append(df_ds_temp, ignore_index=True)
        #
        df_ds.to_json('%s.json'%prefix,orient='columns')



#














#
