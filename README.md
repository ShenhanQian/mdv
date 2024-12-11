# Muti-Dimension Viewer

Navigate hierarchical data in multi dimensions with your keyboard and mouse (wheel).

<div align="center"> 
  <img src="demo.gif">
</div> 

## Data Structure
```
<root_folder>
|- <level0_0>
|       |- <level1_0>
|       |       |- <level2_0>
|       |       |- <level2_1>
|       |       |- ...
|       |
|       |- <level1_1>
|       |- <level1_2>
|       |- ...
|
|- <level0_1>
|- <level0_2>
|- ...
```

Navigate an arbitrary number of levels of folders and files. Current supporte file types:
- images: jpg (jpeg), png

More file types to support:
- meshes/point clouds
- videos

<!-- <div align="center">
        <image src="./screenshot.png" height=800px></image>
</div> -->

## Installation

```shell
pip install git+https://github.com/ShenhanQian/mdv.git
```

## Usage

```shell
mdv <root_folder>
```

Run with`-h` to see all arguments.
