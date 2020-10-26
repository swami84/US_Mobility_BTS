import folium
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import json
import geopandas as gpd
from src.utils import collect_data
from branca.element import MacroElement
from jinja2 import Template
import numpy as np
import matplotlib
from matplotlib.colors import rgb2hex
from branca.colormap import LinearColormap
import datetime
import os
from random import randint

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
        DataAnalysis()

    def plot_metric(self,df_mob, var_list, rolling_mean=False, df_mob_rm=None,auto_y_lim=False):
        f, axes = plt.subplots(len(var_list), 1, sharex=True, figsize=(5 * len(var_list), 20))
        if rolling_mean:

            for var, ax in zip(var_list, axes.flatten()):
                df_gr = df_mob.groupby(['fips', 'date'])[var].mean().unstack(level=0)
                df_gr_rm = df_mob_rm.groupby(['fips', 'date'])[var].mean().unstack(level=0)
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
    def generate_geojson_map(self,df, col_name, tooltip_col=None, name=None):
        df = df.rename(columns={'fips': 'GEO_ID'})
        mdict = df.set_index('GEO_ID')[col_name].to_dict()

        fpath = self.create_geo_json(df, col_name)
        with open(fpath, encoding="ISO-8859-1") as json_file:
            county_geo = json.load(json_file)
        gmap = folium.GeoJson(
            data=county_geo, name=name,
            tooltip=folium.GeoJsonTooltip(fields=[tooltip_col, col_name]),
            style_function=lambda feature: {
                'fillColor': self.get_color(mdict, feature),
                'fillOpacity': 0.7,
                'color': 'black',
                'weight': 0.5,
            }
        )
        color = plt.get_cmap("plasma")

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

    def get_color(self,map_dict, feature):
        color = plt.get_cmap("plasma")
        min_val = min(map_dict.values())
        max_val = max(map_dict.values())
        if isinstance(min_val, datetime.datetime):
            color_sc = 255 / (max_val - min_val).days
            value = int((map_dict.get(feature['properties']['GEO_ID']) - (min_val)).days * color_sc)
        else:
            color_sc = 255 / (abs(min_val) + abs(max_val))
            value = int((map_dict.get(feature['properties']['GEO_ID']) + abs(min_val)) * color_sc)
        rgb = color.colors[value][:3]
        return str((matplotlib.colors.rgb2hex(rgb)))
    def plot_map(self, col_list,df,save_op = False):
        us_cen = [43.8283, -98.5795]
        base_map = folium.Map(location=us_cen, zoom_start=4)
        maps, cs = [], []
        for col in col_list:
            gmap, color_scale = self.generate_geojson_map(df, col, name=col, tooltip_col='loc')
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
            output_path = './data/output'
            os.makedirs(output_path,exist_ok=True)
            curr_date = str((datetime.datetime.today()).date())
            rand_int = str(randint(0,100000000))
            fpath = output_path + '/' + curr_date + '_' + rand_int + '.html'
            print('Saving file at ', fpath)
            base_map.save(fpath)
        return base_map