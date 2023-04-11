import locale
'''
本地化方式如下所示，此文件是"gui.py"的文本内容
'''

LANGUAGE_LIST = ['zh_CN', 'en_US']
LANGUAGE_ALL = {
    'zh_CN': {
        'SUPER': 'END',
        'LANGUAGE': 'zh_CN',
        '加载模型/Load Model': '加载模型',
        '选择.pth文件/.pth File': '选择.pth文件',
        '选择.index文件/.index File': '选择.index文件',
        '选择.npy文件/.npy File': '选择.npy文件',
        "输入设备/Input Device": "输入设备",
        "输出设备/Output Device": "输出设备",
        '音频设备(请使用同种类驱动)/Audio Devices': '音频设备(请使用同种类驱动)',
        '响应阈值/Silence Threhold': '响应阈值',
        '音调设置/Pitch Offset': '音调设置',
        '常规设置/Common': '常规设置',
        '采样长度/Sample Length': '采样长度',
        '淡入淡出长度/Crossfade Length': '淡入淡出长度',
        '额外推理时长/Extra Length': '额外推理时长',
        '性能设置/Performance': '性能设置',
        '开始音频转换': '开始音频转换',
        '停止音频转换': '停止音频转换'
    },
    'en_US': {
        'SUPER': 'zh_CN',
        'LANGUAGE': 'en_US',
        '加载模型/Load Model': 'Load Model',
        '选择.pth文件/.pth File': 'Select .pth File',
        '选择.index文件/.index File': 'Select .index File',
        '选择.npy文件/.npy File': 'Select .npy File',
        "输入设备/Input Device": "Input Device",
        "输出设备/Output Device": "Output Device",
        '音频设备(请使用同种类驱动)/Audio Devices': 'Audio Devices(Please use the same type of driver)',
        '响应阈值/Silence Threhold': 'Silence Threhold',
        '音调设置/Pitch Offset': 'Pitch Offset',
        '常规设置/Common': 'Common',
        '采样长度/Sample Length': 'Sample Length',
        '淡入淡出长度/Crossfade Length': 'Crossfade Length',
        '额外推理时长/Extra Length': 'Extra Length',
        '性能设置/Performance': 'Performance',
        '开始音频转换': 'Start VC',
        '停止音频转换': 'Stop VC'
    }
}


class I18nAuto:
    def __init__(self, language=None):
        self.language_list = LANGUAGE_LIST
        self.language_all = LANGUAGE_ALL
        self.language_map = {}
        if language is None:
            language = 'auto'
        if language == 'auto':
            language = locale.getdefaultlocale()[0]
            if language not in self.language_list:
                language = 'zh_CN'
        self.language = language
        super_language_list = []
        while self.language_all[language]['SUPER'] != 'END':
            super_language_list.append(language)
            language = self.language_all[language]['SUPER']
        super_language_list.append('zh_CN')
        super_language_list.reverse()
        for _lang in super_language_list:
            self.read_language(self.language_all[_lang])

    def read_language(self, lang_dict: dict):
        for _key in lang_dict.keys():
            self.language_map[_key] = lang_dict[_key]

    def __call__(self, key):
        return self.language_map[key]
