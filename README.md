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
- images: jpg (jpeg), png, flo5, dsp5
- meshes: obj, glb, ply
- containers: npz

More file types to support:
- videos
- point cloud sequence

More functionalities to come:
- zoomin/out for images

## Installation

```shell
pip install git+https://github.com/ShenhanQian/mdv.git
```

## Usage

```shell
mdv <root_folder>
```

Run with`-h` to see all arguments.

### Mesh Viewer
|Key|Action                   |
| - | -                       |
|`w`|camera move forward      |
|`s`|camera move backward     |
|`a`|camera move left         |
|`d`|camera move right        |
|`e`|camera move up           |
|`q`|camera move down         |
|`r`|reset camera pose        |
|`[`|decrease light intensity |
|`]`|increase light intensity |

## Dependancy
- DearPyGUI
- PyRender 
