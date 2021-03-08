import folium
import matplotlib.pyplot as plt
import pandas as pd
import json
import geopandas as gpd
from branca.element import MacroElement
from jinja2 import Template
import numpy as np
import matplotlib
from matplotlib.colors import rgb2hex
from branca.colormap import LinearColormap
import datetime
import os
from random import randint
from folium.plugins import TimeSliderChoropleth


# Create a class to bind color maps to respective geojson maps


class BindColormap(MacroElement):
    """Binds a colormap to a given layer.

    Parameters
    ----------
    colormap : branca.colormap.ColorMap
        The colormap to bind.
    """
    def __init__(self, layer, colormap):
        super(BindColormap, self).__init__()
        self.layer = layer
        self.colormap = colormap
        self._template = Template(u"""
        {% macro script(this, kwargs) %}
            {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
            {{this._parent.get_name()}}.on('overlayadd', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
                }});
            {{this._parent.get_name()}}.on('overlayremove', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'none';
                }});
        {% endmacro %}
        """)  # noqa

class DataAnalysis():
    if __name__ == '__main__':
        print('Data Analysis')

    def save_fig(self,fname):
        output_path = './data/output/images/'
        os.makedirs(output_path, exist_ok=True)
        filename = output_path + fname + '.jpg'
        plt.savefig(filename, bbox_inches='tight')


    def plot_metric(self,df_mob, var_list, rolling_mean=False, df_mob_rm=None,auto_y_lim=False):
        f, axes = plt.subplots(len(var_list), 1, sharex=True, figsize=(5 * len(var_list), 20))
        if rolling_mean:

            for var, ax in zip(var_list, axes.flatten()):
                df_gr = df_mob.groupby(['fips', 'date'])[var].mean().unstack(level=0)
                df_gr_rm = df_mob_rm.groupby(['fips', 'date'])[var].mean().unstack(level=0)
                if var == 'Number of Trips':
                    ax.plot(df_gr.index, df_gr.mean(axis=1).values, label='Daily Data')
                    ax.fill_between(df_gr.index, df_gr.mean(axis=1) - 0.1 * df_gr.std(axis=1),
                                    df_gr.mean(axis=1) + 0.1 * df_gr.std(axis=1), alpha=0.2)
                    ax.plot(df_gr_rm.index, df_gr_rm.mean(axis=1).values, color='red', label='7-day Moving Average')
                    ax.fill_between(df_gr_rm.index, df_gr_rm.mean(axis=1) - 0.1 * df_gr_rm.std(axis=1),
                                    df_gr_rm.mean(axis=1) + 0.1 * df_gr_rm.std(axis=1), alpha=0.2, color='red')
                else:
                    ax.plot(df_gr.index, df_gr.mean(axis=1).values, label='Daily Data')
                    ax.fill_between(df_gr.index, df_gr.mean(axis=1) - 0.5 * df_gr.std(axis=1),
                                    df_gr.mean(axis=1) + 0.5 * df_gr.std(axis=1), alpha=0.2)
                    ax.plot(df_gr_rm.index, df_gr_rm.mean(axis=1).values, color='red', label='7-day Moving Average')
                    ax.fill_between(df_gr_rm.index, df_gr_rm.mean(axis=1) - 0.5 * df_gr_rm.std(axis=1),
                                    df_gr_rm.mean(axis=1) + 0.5 * df_gr_rm.std(axis=1), alpha=0.2, color='red')

                if not auto_y_lim:
                    ax.set_ylim(df_gr.mean(axis=1).mean() - 5 * df_gr_rm.mean(axis=1).std(),
                                df_gr_rm.mean(axis=1).mean() + 5 * df_gr_rm.mean(axis=1).std())

                ax.set_title(var, fontsize=18)
                ax.set_xlabel('Date', fontsize=16)
                ax.set_ylabel(var, fontsize=16)
                if np.where(axes == ax)[0][0] == (len(axes) - 1):
                    ax.legend(fontsize=18)

        else:
            for var, ax in zip(var_list, axes.flatten()):
                df_gr = df_mob.groupby(['fips', 'date'])[var].mean().unstack(level=0)
                ax.plot(df_gr.index, df_gr.mean(axis=1).values)
                ax.fill_between(df_gr.index, df_gr.mean(axis=1) - 0.5 * df_gr.std(axis=1),
                                df_gr.mean(axis=1) + 0.5 * df_gr.std(axis=1), alpha=0.5)
                ax.set_ylim(df_gr.mean(axis=1).mean() - 5 * df_gr.mean(axis=1).std(),
                            df_gr.mean(axis=1).mean() + 5 * df_gr.mean(axis=1).std())

    # Creating functions to create map/json and color scales
    def get_county_state(self,val_dict, x):
        return val_dict[x]
    def get_df_loc(self,df):
        df_county_geocodes = pd.read_csv('./Data/County_GEOCODES-v2017.csv', encoding='latin')
        df_county_geocodes['fips'] = df_county_geocodes['fips'].astype(str).str.zfill(5)
        county_dict = df_county_geocodes.set_index('fips')['long_name'].to_dict()
        df['loc'] = df['fips'].apply(lambda x: self.get_county_state(county_dict, x))
        return df
    def generate_geojson_map(self,df, col_name,cmap,q_filter,filter_level, tooltip_col=None, name=None):
        if 'loc' not in df.columns:
            df = self.get_df_loc(df)
        df = df.rename(columns={'fips': 'GEO_ID'})
        if q_filter:
            if filter_level:
                low_fil = filter_level[0]
                high_fil = filter_level[1]
                df.loc[df[col_name] > df[col_name].quantile(high_fil), col_name] = df[col_name].quantile(high_fil)
                df.loc[df[col_name] < df[col_name].quantile(low_fil), col_name] = df[col_name].quantile(low_fil)
            else:
                print('No outlier filtering done. Please filter levels as a list')

        mdict = df.set_index('GEO_ID')[col_name].to_dict()

        fpath = self.create_geo_json(df, col_name)
        with open(fpath, encoding="ISO-8859-1") as json_file:
            county_geo = json.load(json_file)
        gmap = folium.GeoJson(
            data=county_geo, name=name,overlay=True,
            tooltip=folium.GeoJsonTooltip(fields=[tooltip_col, col_name]),
            style_function=lambda feature: {
                'fillColor': self.get_color(mdict, feature,cmap),
                'fillOpacity': 0.7,
                'color': 'black',
                'weight': 0.5,
            }
        )
        color = plt.get_cmap(cmap)
        if isinstance(color, matplotlib.colors.LinearSegmentedColormap):
            min_color = rgb2hex(color(0)[:3])
            mid_color = rgb2hex(color(127)[:3])
            max_color = rgb2hex(color(256)[:3])
        else:
            min_color = rgb2hex(color.colors[0][:3])
            mid_color = rgb2hex(color.colors[127][:3])
            max_color = rgb2hex(color.colors[-1][:3])
        if isinstance(min(mdict.values()),datetime.datetime):
            min_val = 0
            max_val = (max(mdict.values()) - min(mdict.values())).days
            color_scale = LinearColormap([min_color, mid_color, max_color],
                                         vmin=min_val,
                                         vmax=max_val)
        else:
            color_scale = LinearColormap([min_color, mid_color, max_color],
                                     vmin=min(mdict.values()),
                                     vmax=max(mdict.values()))
        os.remove(fpath)
        return gmap, color_scale

    def filter_geo(self,geo_list, x):
        if x in geo_list:
            return x
        else:
            return None

    def get_col_value(self,val_dict, x):
        return val_dict[x]

    def create_geo_json(self,df, col_name):
        geo_list = df.GEO_ID.unique()
        df = df[[col_name, 'GEO_ID', 'loc']]
        
        df_geo = gpd.read_file('./Data/geometry/county.geojson', driver='GeoJSON')
        df_geo['GEO_ID'] = df_geo['GEO_ID'].astype(str).str[-5:]
        df_geo['GEO_ID'] = df_geo['GEO_ID'].apply(lambda x: self.filter_geo(geo_list, x))
        df_geo = df_geo.dropna()
        df_geo = df_geo.join(df.set_index('GEO_ID'), on='GEO_ID', how='inner')
        df_geo = df_geo.dropna()
        fpath = './Data/geometry/County_Geo_' + col_name + '.json'
        fpath = fpath.replace('<', '_less_than_')
        fpath = fpath.replace('>', '_greater_than_')
        df_geo.to_file(fpath, driver='GeoJSON')

        return (fpath)

    def get_color(self,map_dict, feature,cmap):
        color = plt.get_cmap(cmap)
        min_val = min(map_dict.values())
        max_val = max(map_dict.values())
        if isinstance(min_val, datetime.datetime):
            color_sc = 255 / (max_val - min_val).days
            value = int((map_dict.get(feature['properties']['GEO_ID']) - (min_val)).days * color_sc)
        else:
            color_sc = 255 / (abs(min_val) + abs(max_val))
            value = int((map_dict.get(feature['properties']['GEO_ID']) + abs(min_val)) * color_sc)
        if isinstance(color, matplotlib.colors.LinearSegmentedColormap):
            rgb = color(value)[:3]
        else:
            rgb = color.colors[value][:3]
        return str((matplotlib.colors.rgb2hex(rgb)))
    def plot_map(self, col_list,df,save_op = False,cmap='plasma',q_filter=False,filter_level=[0.01,0.99],name=None):
        us_cen = [43.8283, -98.5795]
        base_map = folium.Map(location=us_cen, zoom_start=4)
        maps, cs = [], []
        for col in col_list:
            gmap, color_scale = self.generate_geojson_map(df, col, name=col, tooltip_col='loc',cmap=cmap,
                                                          q_filter=q_filter,filter_level=filter_level)
            maps.append(gmap)
            cs.append(color_scale)
        for m in maps:
            base_map.add_child(m)
        base_map.add_child(folium.map.LayerControl())
        for col_sc in cs:
            base_map.add_child(col_sc)
        for m, col_sc in zip(maps, cs):
            base_map.add_child(BindColormap(m, col_sc))
        if save_op:
            output_path = './data/output/maps/'
            os.makedirs(output_path,exist_ok=True)
            curr_date = str((datetime.datetime.today()).date())
            fpath = output_path + '/' + curr_date +'_' + name +'.html'
            print('Saving file at ', fpath)
            base_map.save(fpath)
            from selenium import webdriver
            path = "../../webdrivers/chromedriver.exe"
            driver = webdriver.Chrome(path)
            html_path = 'file:///C:/Work/projects/mobility/US_Mobility_BTS/Data/output/maps/'+curr_date +'_' + name +'.html'
            driver.get(html_path);
            fname = './data/output/images/' + curr_date + '_' + name + '.jpg'
            driver.save_screenshot(fname);
            driver.quit();
        return base_map


    def get_style_dict(self,df, df_geo):
        def get_state_id(x, sdict):
            if x in sdict.keys():
                return sdict[x]
            else:
                return None

        sdict = df_geo.reset_index().set_index('GEO_ID')['index'].to_dict()

        df['id'] = (df['fips'].apply(lambda x: get_state_id(x, sdict)))
        df = df.dropna()
        df['id'] = df['id'].astype(int)
        styledata = {}
        for st in df.id.unique():
            df_st = df[df.id == st][['date', 'color', 'opacity']].sort_values('date').set_index('date')
            styledata[st] = df_st
        styledict = {
            str(country): data.to_dict(orient='index') for
            country, data in styledata.items()}
        return styledict

    def generate_ts_geo(self,df_geo,styledict, df, targ_var, name):

        g = TimeSliderChoropleth(
            df_geo.to_json(),
            styledict=styledict, name=name)
        color = plt.get_cmap("plasma")
        min_color = rgb2hex(color.colors[0][:3])
        mid_color = rgb2hex(color.colors[127][:3])
        max_color = rgb2hex(color.colors[-1][:3])

        min_val = (df[targ_var].min())
        max_val = (df[targ_var].max())
        color_scale = LinearColormap([min_color, mid_color, max_color],
                                     vmin=min_val,
                                     vmax=max_val)
        return g, color_scale

    def get_rgb_col(self,x):
        import matplotlib
        color = plt.get_cmap("plasma")
        rgb = color.colors[x][:3]
        return str(matplotlib.colors.rgb2hex(rgb))

    def group_df(self,df,targ_var,q_filter):
        df_gr = df.groupby(['fips', 'month'])[targ_var].mean().reset_index()
        df_gr = df_gr.fillna(df_gr.mean())
        if q_filter:
            df_gr.loc[df_gr[targ_var] > df_gr[targ_var].quantile(0.99), targ_var] = df_gr[targ_var].quantile(0.99)

        min_val = (df_gr[targ_var].min())
        max_val = (df_gr[targ_var].max())
        col_scl = 255 / (abs(min_val) + abs(max_val))
        df_gr['color'] = (((df_gr[targ_var]) - abs(min_val)) * col_scl).astype(int)
        df_gr['color'] = df_gr['color'].apply(lambda x: self.get_rgb_col(x))
        df_gr['date'] = df_gr["month"].apply(lambda x: datetime.datetime.strptime('2019-' + str(x) + '-02', "%Y-%B-%d"))
        df_gr.date = ((df_gr['date'].astype('int64')) // 1e9).astype('U10')

        df_gr['opacity'] = 0.75
        return df_gr
    def plot_ts_map(self,df,targ_var,q_filter):
        df_geo = gpd.read_file('./Data/geometry/county.geojson', driver='GeoJSON')

        df_geo['GEO_ID'] = df_geo['GEO_ID'].astype(str).str[-5:]
        df_gr = self.group_df(df,targ_var,q_filter)
        styledict = self.get_style_dict(df_gr, df_geo)

        g, color_scale = self.generate_ts_geo( df_geo,styledict, df_gr,targ_var,  name=targ_var)
        us_cen = [43.8283, -98.5795]

        base_map = folium.Map(location=us_cen, zoom_start=4, tiles='cartodbpositron')
        base_map.add_child(g)
        base_map.add_child(folium.map.LayerControl())

        base_map.add_child(color_scale)
        base_map.add_child(BindColormap(g, color_scale))


        return base_map

