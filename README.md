Tools for mapping boring locations, computing sandy-layer percentages, and plotting SPT logs by building.

To run the data;

map_and_logs.py --intervals spt_intervals_sunnyisles_full.csv --locations boring_locations.csv --outdir geosec_out --nscale 60

It will create;
Map: geosec_out/sunnyisles_borings_map.html
Summary: geosec_out/sandy_percentages_by_boring.csv
Output dir: geosec_out
