# Muti-Dimension Viewer

Navigate hierarchical data in multi dimensions with your keyboard and mouse (wheel).

<div align="center"> 
  <img src="demo.gif">
</div> 

## Data Structure
```
<root_folder>
|- <row_0>
|       |- <col_0>
|       |- <col_1>
|       |- ...
|
|- <row_1>
|- <row_2>
|- ...
```

Current supported file type: 
- images

More file types to support:
- videos
- meshes/point clouds

More scenarios:
- `<col>` is in a deeper level of `<row>`
- More than two levels

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
