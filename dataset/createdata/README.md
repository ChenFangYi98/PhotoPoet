# PhotoPoet 数据制作

本目录包含下载、标注、审核、分片导出与验证脚本。`PhotoPoet-Data/` 由运行时创建，数据分片和来源记录见独立数据目录。

```powershell
python dataset/createdata/photopoet_run.py --help
python dataset/createdata/photopoet_verify.py
```

正式运行会下载公开数据与教师模型并写入本地工作目录。先阅读脚本配置，核对来源、磁盘空间和目标行数。
