这是我当前的文章的一部分

```
\subsection{基本流程}
\label{sec:method_basic}

图二\footnote{TODO: 待画图}\footnote{subsectiond “基本流程”也可以修改为“overview”}展示了ViT-MGI算法的整个流程。首先，ViT-MGI收集各个客户端上传上来的梯度变化结果，然后根据\hyperref[sec:exp_layer]{§\ref{sec:exp_layer}}得出的结论将攻击识别所关注的层提取出来。如图三\footnote{TODO: 待画图：特征层提取}所示，对于某个客户端$c_i$，我们将其上传到中央服务器的梯度记为$g_i$，ViT-MGI首先将$g_i$中的mtta.hidden层、jsj.iiuio层\footnote{待确定}等全部提取出来重新拼接到一个向量中并记为$\phi(g_i)$，之后使用主成分分析算法对这些数据中的主成分进行分析。

主成分分析
```

请你帮忙续写主成分分析的这一部分。







先参考这段话简洁一点，先写一段话，后面有专门详细介绍这个的地方。







这部分我已经写过了

```
图二\footnote{TODO: 待画图}\footnote{subsectiond “基本流程”也可以修改为“overview”}展示了ViT-MGI算法的整个流程。首先，ViT-MGI收集各个客户端上传上来的梯度变化结果，然后根据\hyperref[sec:exp_layer]{§\ref{sec:exp_layer}}得出的结论将攻击识别所关注的层提取出来。如图三\footnote{TODO: 待画图：特征层提取}所示，对于某个客户端$c_i$，我们将其上传到中央服务器的梯度记为$g_i$，ViT-MGI首先将$g_i$中的mtta.hidden层、jsj.iiuio层\footnote{待确定}等全部提取出来重新拼接到一个向量中并记为$\phi(g_i)$，之后使用主成分分析算法对这些数据中的主成分进行分析。
```

这这部分的基础上，先续写一段简单的介绍主成分分析的内容。






再尝试使用更简短的话介绍一下主成分分析的原理






Latex如何添加代码块





algorithmic是干什么的






很棒，详细介绍一下algorithm和algorithmic怎么使用







\REQUIRE和\ENSURE能否替换成input和output





\begin{algorithm}
    \caption{特征层提取}
    \label{alg:example}
    \begin{algorithmic}
        \REQUIRE $g_i$ - 第$i$个客户端上传的梯度; $L_{keep}$ - 要保留的层
        \ENSURE $\phi(g_i)$ - 提取特征层后的梯度
        \FOR{$l\in g_i.layers()$}
            \IF $l.name\in L_{keep}$
                \STATE $\phi(l)\gets l$
            \ENDIF
            \STATE $j \gets i - 1$
            \WHILE{$j \geq 0$ 且 $A[j] > key$}
                \STATE $A[j+1] \gets A[j]$
                \STATE $j \gets j - 1$
                \COMMENT{向左移动元素}
            \ENDWHILE
            \STATE $A[j+1] \gets key$
        \ENDFOR
    \end{algorithmic}
\end{algorithm}


! Missing $ inserted.
<inserted text> 
                $
l.313             \IF $
                       l.name\in L_{keep}$







如何return





如何使用`function`和`end function`命令





latex EndFor 不换行








\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\renewcommand{\algorithmicrequire}{\textbf{Input:}}
\renewcommand{\algorithmicensure}{\textbf{Output:}}
% \newcommand{\FUNCTION}[1]{\STATE \textbf{function} \textsc{#1}}
% \newcommand{\ENDFUNCTION}{\STATE \textbf{end function}}
\usepackage{graphicx}

\begin{algorithm}
    \caption{特征层提取}
    \label{alg:example}
    \begin{algorithmic}[1]
        \Require $g_i$ - 第$i$个客户端上传的梯度; $L_{keep}$ - 要保留的层
        \Ensure $\phi(g_i)$ - 提取特征层后的梯度
        \Function{Extract}{$g_i, L_{keep}$}
        \State $\phi(g_i)\gets\emptyset$
        \For{$l\in g_i.layers()$}
            \If{$l.name\in L_{keep}$}
                \State $\phi(g_i) += l$
            \EndIf
        \EndFor
        \Return $\phi(g_i)$
        \EndFunction
    \end{algorithmic}
\end{algorithm}







这是我特征层提取部分的结果：

```
\subsection{特征层提取}
\label{sec:method_layer}

如图三\footnote{TODO: 待画图：特征层提取}所示，对于某个客户端$c_i$，其上传到中央服务器的梯度变化数量为$85806346$个。将其上传到中央服务器的梯度记为$g_i$，ViT-MGI首先将$g_i$中的mtta.hidden层、jsj.iiuio层\footnote{待确定}等全部提取出来，之后将其重新拼接到一个向量中。我们将这些提取出来的更加有效的数据记为$\phi(g_i)$，则经过特征层提取这一步的操作后，每个客户端的参数数量将由$g_i$的$85806346$个减少到$\phi(g_i)$的$12500$个。

这些特征层是由我们经过一系列实验得到的。在实验\hyperref[sec:exp_layer]{§\ref{sec:exp_layer}}中，我们将ViT模型划分为了$2048$\footnote{待确定}个层，其中每个层大约平均就只有几十万个参数。

% \begin{algorithm}
%     \caption{特征层提取}
%     \label{alg:example}
%     \begin{algorithmic}[1]
%         \Require $g_i$ - 第$i$个客户端上传的梯度; $L_{keep}$ - 要保留的层
%         \Ensure $\phi(g_i)$ - 提取特征层后的梯度
%         \Function{Extract}{$g_i, L_{keep}$}
%         \For{$l\in g_i.layers()$}
%             \If{$l.name\in L_{keep}$}
%                 \State $\phi_i\gets l$
%             \EndIf
%         \EndFor
%         \State $\phi(g_i)\gets Concatenate(\{\phi_i | i=1, 2, \cdots, len(L_{keep}) \})$
%         \Return $\phi(g_i)$
%         \EndFunction
%     \end{algorithmic}
% \end{algorithm}

\begin{algorithm}
    \caption{特征层提取}
    \label{alg:example}
    \begin{algorithmic}[1]
        \Require $g_i$ - 第$i$个客户端上传的梯度; $L_{keep}$ - 要保留的层
        \Ensure $\phi(g_i)$ - 提取特征层后的梯度
        \Function{Extract}{$g_i, L_{keep}$}
            \State $\phi(g_i)\gets\emptyset$
            \For{$l\in g_i.layers()$}
                \If{$l.name \in L_{keep}$}
                    \State $\phi(g_i) += l$
                \EndIf
            \EndFor
            \State \Return $\phi(g_i)$
        \EndFunction
    \end{algorithmic}
\end{algorithm}
```

接下来我要开始写主成分分析的相关内容，请你写一段。注意参考上面的写法。







接下来请写隔离森林相关的部分





最后请写主观逻辑模型相关的部分






写一个总流程的伪代码





有更好看一点的注释方法吗





好了，现在我要开始写实验部分了。

实验部分的第一部分是特征层的确定，即确定保留哪些层要被保留。

请你帮我完成写作。






如何查看Linux系统的型号、CPU配置、GPU配置等信息？







系统信息查询结果如下：

```
(base)  ltf@admin  /home/lzy/ltf/Codes/FLDefinder/paper   paper  lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 20.04.3 LTS
Release:        20.04
Codename:       focal
(base)  ltf@admin  /home/lzy/ltf/Codes/FLDefinder/paper   paper  cat /etc/os-release
NAME="Ubuntu"
VERSION="20.04.3 LTS (Focal Fossa)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 20.04.3 LTS"
VERSION_ID="20.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=focal
UBUNTU_CODENAME=focal
(base)  ltf@admin  /home/lzy/ltf/Codes/FLDefinder/paper   paper  uname -r
5.15.0-73-generic
(base)  ltf@admin  /home/lzy/ltf/Codes/FLDefinder/paper   paper  sudo dmidecode -t system
# dmidecode 3.2
Getting SMBIOS data from sysfs.
SMBIOS 3.2.0 present.

Handle 0x0001, DMI type 1, 27 bytes
System Information
        Manufacturer: System manufacturer
        Product Name: System Product Name
        Version: System Version
        Serial Number: System Serial Number
        UUID: f8c7e430-98ba-a69d-bf80-04d4c45bc8c4
        Wake-up Type: Power Switch
        SKU Number: ASUS_MB_KBLX
        Family: To be filled by O.E.M.

Handle 0x0032, DMI type 12, 5 bytes
System Configuration Options
        Option 1: SMI:00B29C05
        Option 2: DSN:                               .
        Option 3: DSN:                               .
        Option 4: DSN:                               .

Handle 0x0033, DMI type 32, 20 bytes
System Boot Information
        Status: No errors detected

Handle 0x0049, DMI type 15, 75 bytes
System Event Log
        Area Length: 4095 bytes
        Header Start Offset: 0x0000
        Header Length: 16 bytes
        Data Start Offset: 0x0010
        Access Method: Memory-mapped physical 32-bit address
        Access Address: 0xFF300000
        Status: Valid, Not Full
        Change Token: 0x00000001
        Header Format: Type 1
        Supported Log Type Descriptors: 26
        Descriptor 1: Single-bit ECC memory error
        Data Format 1: Multiple-event handle
        Descriptor 2: Multi-bit ECC memory error
        Data Format 2: Multiple-event handle
        Descriptor 3: Parity memory error
        Data Format 3: None
        Descriptor 4: Bus timeout
        Data Format 4: None
        Descriptor 5: I/O channel block
        Data Format 5: None
        Descriptor 6: Software NMI
        Data Format 6: None
        Descriptor 7: POST memory resize
        Data Format 7: None
        Descriptor 8: POST error
        Data Format 8: POST results bitmap
        Descriptor 9: PCI parity error
        Data Format 9: Multiple-event handle
        Descriptor 10: PCI system error
        Data Format 10: Multiple-event handle
        Descriptor 11: CPU failure
        Data Format 11: None
        Descriptor 12: EISA failsafe timer timeout
        Data Format 12: None
        Descriptor 13: Correctable memory log disabled
        Data Format 13: None
        Descriptor 14: Logging disabled
        Data Format 14: None
        Descriptor 15: System limit exceeded
        Data Format 15: None
        Descriptor 16: Asynchronous hardware timer expired
        Data Format 16: None
        Descriptor 17: System configuration information
        Data Format 17: None
        Descriptor 18: Hard disk information
        Data Format 18: None
        Descriptor 19: System reconfigured
        Data Format 19: None
        Descriptor 20: Uncorrectable CPU-complex error
        Data Format 20: None
        Descriptor 21: Log area reset/cleared
        Data Format 21: None
        Descriptor 22: System boot
        Data Format 22: None
        Descriptor 23: End of log
        Data Format 23: None
        Descriptor 24: OEM-specific
        Data Format 24: OEM-specific
        Descriptor 25: OEM-specific
        Data Format 25: OEM-specific
        Descriptor 26: OEM-specific
        Data Format 26: OEM-specific

(base)  ltf@admin  /home/lzy/ltf/Codes/FLDefinder/paper   paper  lscpu
架构：                           x86_64
CPU 运行模式：                   32-bit, 64-bit
字节序：                         Little Endian
Address sizes:                   46 bits physical, 48 bits virtual
CPU:                             28
在线 CPU 列表：                  0-27
每个核的线程数：                 2
每个座的核数：                   14
座：                             1
NUMA 节点：                      1
厂商 ID：                        GenuineIntel
CPU 系列：                       6
型号：                           85
型号名称：                       Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz
步进：                           7
CPU MHz：                        3300.000
CPU 最大 MHz：                   4800.0000
CPU 最小 MHz：                   1200.0000
BogoMIPS：                       6599.98
虚拟化：                         VT-x
L1d 缓存：                       448 KiB
L1i 缓存：                       448 KiB
L2 缓存：                        14 MiB
L3 缓存：                        19.3 MiB
NUMA 节点0 CPU：                 0-27
Vulnerability Itlb multihit:     KVM: Mitigation: VMX disabled
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Mmio stale data:   Mitigation; Clear CPU buffers; SMT vulnerable
Vulnerability Retbleed:          Mitigation; Enhanced IBRS
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Mitigation; TSX disabled
标记：                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon
                                  pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe p
                                 opcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi f
                                 lexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512v
                                 l xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req avx512_vnni md_clear flush_l1d a
                                 rch_capabilities
(base)  ltf@admin  /home/lzy/ltf/Codes/FLDefinder/paper   paper  free -h
              总计         已用        空闲      共享    缓冲/缓存    可用
内存：       125Gi        17Gi        82Gi       3.0Mi        25Gi       107Gi
交换：       2.0Gi       547Mi       1.5Gi
(base)  ltf@admin  /home/lzy/ltf/Codes/FLDefinder/paper   paper  sudo dmidecode -t memory
# dmidecode 3.2
Getting SMBIOS data from sysfs.
SMBIOS 3.2.0 present.

Handle 0x004A, DMI type 16, 23 bytes
Physical Memory Array
        Location: System Board Or Motherboard
        Use: System Memory
        Error Correction Type: None
        Maximum Capacity: 3 TB
        Error Information Handle: Not Provided
        Number Of Devices: 8

Handle 0x004B, DMI type 17, 40 bytes
Memory Device
        Array Handle: 0x004A
        Error Information Handle: Not Provided
        Total Width: 72 bits
        Data Width: 64 bits
        Size: 32 GB
        Form Factor: DIMM
        Set: None
        Locator: DIMM_A1
        Bank Locator: NODE 1
        Type: DDR4
        Type Detail: Synchronous
        Speed: 2933 MT/s
        Manufacturer: Samsung
        Serial Number: 40B14DE4
        Asset Tag:  
        Part Number: M378A4G43AB2-CWE    
        Rank: 2
        Configured Memory Speed: 2933 MT/s
        Minimum Voltage: 1.2 V
        Maximum Voltage: 1.2 V
        Configured Voltage: 1.2 V

Handle 0x004C, DMI type 17, 40 bytes
Memory Device
        Array Handle: 0x004A
        Error Information Handle: Not Provided
        Total Width: Unknown
        Data Width: Unknown
        Size: No Module Installed
        Form Factor: DIMM
        Set: None
        Locator: DIMM_A2
        Bank Locator: NODE 1
        Type: Unknown
        Type Detail: Synchronous
        Speed: Unknown
        Manufacturer: NO DIMM
        Serial Number: NO DIMM
        Asset Tag:  
        Part Number: NO DIMM
        Rank: Unknown
        Configured Memory Speed: Unknown
        Minimum Voltage: 1.2 V
        Maximum Voltage: 1.2 V
        Configured Voltage: 1.2 V

Handle 0x004D, DMI type 17, 40 bytes
Memory Device
        Array Handle: 0x004A
        Error Information Handle: Not Provided
        Total Width: 72 bits
        Data Width: 64 bits
        Size: 32 GB
        Form Factor: DIMM
        Set: None
        Locator: DIMM_B1
        Bank Locator: NODE 1
        Type: DDR4
        Type Detail: Synchronous
        Speed: 2933 MT/s
        Manufacturer: Samsung
        Serial Number: 40B14DDE
        Asset Tag:  
        Part Number: M378A4G43AB2-CWE    
        Rank: 2
        Configured Memory Speed: 2933 MT/s
        Minimum Voltage: 1.2 V
        Maximum Voltage: 1.2 V
        Configured Voltage: 1.2 V

Handle 0x004E, DMI type 17, 40 bytes
Memory Device
        Array Handle: 0x004A
        Error Information Handle: Not Provided
        Total Width: Unknown
        Data Width: Unknown
        Size: No Module Installed
        Form Factor: DIMM
        Set: None
        Locator: DIMM_B2
        Bank Locator: NODE 1
        Type: Unknown
        Type Detail: Synchronous
        Speed: Unknown
        Manufacturer: NO DIMM
        Serial Number: NO DIMM
        Asset Tag:  
        Part Number: NO DIMM
        Rank: Unknown
        Configured Memory Speed: Unknown
        Minimum Voltage: 1.2 V
        Maximum Voltage: 1.2 V
        Configured Voltage: 1.2 V

Handle 0x004F, DMI type 17, 40 bytes
Memory Device
        Array Handle: 0x004A
        Error Information Handle: Not Provided
        Total Width: 72 bits
        Data Width: 64 bits
        Size: 32 GB
        Form Factor: DIMM
        Set: None
        Locator: DIMM_C1
        Bank Locator: NODE 1
        Type: DDR4
        Type Detail: Synchronous
        Speed: 2933 MT/s
        Manufacturer: Samsung
        Serial Number: 40B1682F
        Asset Tag:  
        Part Number: M378A4G43AB2-CWE    
        Rank: 2
        Configured Memory Speed: 2933 MT/s
        Minimum Voltage: 1.2 V
        Maximum Voltage: 1.2 V
        Configured Voltage: 1.2 V

Handle 0x0050, DMI type 17, 40 bytes
Memory Device
        Array Handle: 0x004A
        Error Information Handle: Not Provided
        Total Width: Unknown
        Data Width: Unknown
        Size: No Module Installed
        Form Factor: DIMM
        Set: None
        Locator: DIMM_C2
        Bank Locator: NODE 1
        Type: Unknown
        Type Detail: Synchronous
        Speed: Unknown
        Manufacturer: NO DIMM
        Serial Number: NO DIMM
        Asset Tag:  
        Part Number: NO DIMM
        Rank: Unknown
        Configured Memory Speed: Unknown
        Minimum Voltage: 1.2 V
        Maximum Voltage: 1.2 V
        Configured Voltage: 1.2 V

Handle 0x0051, DMI type 17, 40 bytes
Memory Device
        Array Handle: 0x004A
        Error Information Handle: Not Provided
        Total Width: 72 bits
        Data Width: 64 bits
        Size: 32 GB
        Form Factor: DIMM
        Set: None
        Locator: DIMM_D1
        Bank Locator: NODE 1
        Type: DDR4
        Type Detail: Synchronous
        Speed: 2933 MT/s
        Manufacturer: Samsung
        Serial Number: 40B14E42
        Asset Tag:  
        Part Number: M378A4G43AB2-CWE    
        Rank: 2
        Configured Memory Speed: 2933 MT/s
        Minimum Voltage: 1.2 V
        Maximum Voltage: 1.2 V
        Configured Voltage: 1.2 V

Handle 0x0052, DMI type 17, 40 bytes
Memory Device
        Array Handle: 0x004A
        Error Information Handle: Not Provided
        Total Width: Unknown
        Data Width: Unknown
        Size: No Module Installed
        Form Factor: DIMM
        Set: None
        Locator: DIMM_D2
        Bank Locator: NODE 1
        Type: Unknown
        Type Detail: Synchronous
        Speed: Unknown
        Manufacturer: NO DIMM
        Serial Number: NO DIMM
        Asset Tag:  
        Part Number: NO DIMM
        Rank: Unknown
        Configured Memory Speed: Unknown
        Minimum Voltage: 1.2 V
        Maximum Voltage: 1.2 V
        Configured Voltage: 1.2 V

(base)  ltf@admin  /home/lzy/ltf/Codes/FLDefinder/paper   paper  nvidia-smi
Thu Jul 11 15:54:23 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.239.06   Driver Version: 470.239.06   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:1A:00.0 Off |                  N/A |
|  0%   47C    P8    20W / 370W |      5MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  On   | 00000000:68:00.0 Off |                  N/A |
|  0%   50C    P8    29W / 370W |     19MiB / 24265MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2715      G   /usr/lib/xorg/Xorg                  4MiB |
|    1   N/A  N/A      2715      G   /usr/lib/xorg/Xorg                  9MiB |
|    1   N/A  N/A      2767      G   /usr/bin/gnome-shell                8MiB |
+-----------------------------------------------------------------------------+
(base)  ltf@admin  /home/lzy/ltf/Codes/FLDefinder/paper   paper  lspci | grep -i vga
1a:00.0 VGA compatible controller: NVIDIA Corporation Device 2204 (rev a1)
68:00.0 VGA compatible controller: NVIDIA Corporation Device 2204 (rev a1)
```

我现在要开始写实验部分的实验设置，说明我在一台什么样的机器上跑的实验。请你帮忙完成，不需要过长的篇幅。如果你还有哪些需要的信息，可以向我提问。







ViT有哪些常见的模型





除了CIFAR-10，还有哪些论文中常见的计算机视觉数据集？






请续写：

```
\subsection{攻击与防御实验}
\label{exp:attack_defense}

为了能够更好地进一步实验，我们决定首先搭建好联邦学习训练ViT的框架。我们使用PyTorch作为
```





前面已经写过系统配置信息了。

请直接开始写攻击与防御实验。注意，你正在写论文，注意论文的格式，最好用较为官方的话描述，尽量避免“仅仅抛出一些列表”的方式。





ubuntu根据进程id查看程序执行路径





如何查看进程的工作路径









请你从中提取出32次准确率：

```
+- /home/lzy/ltf/Codes/FLDefinder/master/src/utils/MyTimer.py:30 -+
| TimeList:                                                       |
| 00: Start | 2024.07.11-19:54:32                                 |
| 01: init accuracy: 6.50% | 2024.07.11-19:54:40                  |
| 02: Round 1's accuracy: 10.50% | 2024.07.11-19:54:46            |
| 03: Round 2's accuracy: 20.60% | 2024.07.11-19:54:51            |
| 04: Round 3's accuracy: 31.30% | 2024.07.11-19:54:56            |
| 05: Round 4's accuracy: 43.60% | 2024.07.11-19:55:01            |
| 06: Round 5's accuracy: 56.60% | 2024.07.11-19:55:07            |
| 07: Round 6's accuracy: 65.40% | 2024.07.11-19:55:12            |
| 08: Round 7's accuracy: 69.20% | 2024.07.11-19:55:17            |
| 09: Round 8's accuracy: 74.60% | 2024.07.11-19:55:23            |
| 10: Round 9's accuracy: 80.90% | 2024.07.11-19:55:28            |
| 11: Round 10's accuracy: 82.40% | 2024.07.11-19:55:33           |
| 12: Round 11's accuracy: 86.50% | 2024.07.11-19:55:39           |
| 13: Round 12's accuracy: 87.30% | 2024.07.11-19:55:44           |
| 14: Round 13's accuracy: 87.70% | 2024.07.11-19:55:49           |
| 15: Round 14's accuracy: 88.00% | 2024.07.11-19:55:55           |
| 16: Round 15's accuracy: 91.30% | 2024.07.11-19:56:00           |
| 17: Round 16's accuracy: 91.30% | 2024.07.11-19:56:05           |
| 18: Round 17's accuracy: 92.00% | 2024.07.11-19:56:11           |
| 19: Round 18's accuracy: 93.40% | 2024.07.11-19:56:16           |
| 20: Round 19's accuracy: 93.30% | 2024.07.11-19:56:21           |
| 21: Round 20's accuracy: 93.10% | 2024.07.11-19:56:27           |
| 22: Round 21's accuracy: 94.50% | 2024.07.11-19:56:32           |
| 23: Round 22's accuracy: 93.30% | 2024.07.11-19:56:37           |
| 24: Round 23's accuracy: 93.80% | 2024.07.11-19:56:42           |
| 25: Round 24's accuracy: 95.10% | 2024.07.11-19:56:48           |
| 26: Round 25's accuracy: 93.20% | 2024.07.11-19:56:53           |
| 27: Round 26's accuracy: 95.50% | 2024.07.11-19:56:58           |
| 28: Round 27's accuracy: 93.90% | 2024.07.11-19:57:04           |
| 29: Round 28's accuracy: 95.40% | 2024.07.11-19:57:09           |
| 30: Round 29's accuracy: 95.90% | 2024.07.11-19:57:14           |
| 31: Round 30's accuracy: 95.10% | 2024.07.11-19:57:20           |
| 32: Round 31's accuracy: 95.00% | 2024.07.11-19:57:25           |
| 33: Round 32's accuracy: 96.30% | 2024.07.11-19:57:30           |
+-----------------------------------------------------------------+
```






我有5个实验，每个实验有32个数据。我画一张图使得实验结果更加清晰。

也就是说，我有5个长度为32的浮点数数组accuracies0、accuracies1、...，我希望横轴是训练轮次，纵轴是准确率，图例是攻击力度。

我应该怎么画？







! LaTeX Error: Cannot determine size of graphic in pics/001-gradAttack-attackRa
te.svg (no BoundingBox).

See the LaTeX manual or LaTeX Companion for explanation.
Type  H <return>  for immediate help.
 ...                                              
                                                  
l.611 ...hics{pics/001-gradAttack-attackRate.svg}}
                                                  
? 






我\usepackage{svg}了，但是还：

```
\begin{figure}[htbp]
    % \centerline{\includegraphics{pics/001-gradAttack-attackRate.svg}}
    \centering
    \includesvg[width=0.5\textwidth]{pics/001-gradAttack-attackRate}
    \caption{grad ascent attack}
    \label{fig_gradAscent}
\end{figure}
```

```
Package svg Warning: You didn't enable `shell escape' (or `write18')
(svg)                so it wasn't possible to launch the Inkscape export
(svg)                for `pics/001-gradAttack-attackRate.svg' on input line 614
.


! Package svg Error: File `001-gradAttack-attackRate_svg-tex.pdf' is missing.

See the svg package documentation for explanation.
Type  H <return>  for immediate help.
 ...                                              
                                                  
l.614 ...extwidth]{pics/001-gradAttack-attackRate}
                                                  
? 
```






我安装好了`inkscape`，请告诉我完整的编译流程。

注意，我还有references需要编译。







我在overleaf上怎么编译？







我现在决定不使用SVG图片了，而决定使用`pdf`图片。

```
\begin{figure}[htbp]
    \centerline{\includegraphics{pics/001-gradAttack-attackRate.pdf}}
    \caption{grad ascent attack}
    \label{fig_gradAscent}
\end{figure}
```

这段代码导致PDF图片特别大，直接超出了边界。我应该怎么写？







latex如何定义变量？





为什么这段代码引用到的不是图1而是subsection

```
之后，我们在此基础上分别进行了不加任何防御的梯度上升攻击、标签翻转攻击以及后门植入攻击，结果发现对于梯度上升攻击，若不进行防御，则在20\%的攻击者与较小的攻击力度的情况下，模型准确率上升明显减慢。如图\hyperref[fig:gradAscent]{\ref{fig:gradAscent}}所示，攻击者的比例都是20\%。当攻击者把本地训练得到的梯度变化进行取反操作时，发现模型准确率的上升有一定程度的减缓；当攻击者把本地的梯度变化乘以$-2$后上传时，发现模型准确率的上升速率进一步下降；而当攻击者把本地梯度变化乘以$-3$再上传时，可以发现模型已经无法正常工作。

\begin{figure}[htbp]
    \label{fig:gradAscent}
    \centerline{\includegraphics[width=\figGradAscentAttack]{pics/001-gradAttack-attackRate.pdf}}
    % \centering
    % \includesvg[width=0.5\textwidth]{pics/001-gradAttack-attackRate.pdf}
    \caption{grad ascent attack}
\end{figure}
```





latex如何加粗