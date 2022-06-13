from app0 import models
from django import forms
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError
from app0.utils.bootstrap import BootStrapModelForm



# class UserModelForm(forms.ModelForm):

#     # 想做更多的校验,可以重写一下验证规则
#     name = forms.CharField(min_length=2, label="姓名")

#     class Meta:
#         model = models.UserInfo
#         fields = ["name","password","age","account","create_time","gender","depart"]
#         # widgets = {
#         #     "name": forms.TextInput(attrs={"class": "form-control"}),
#         #     "password": forms.TextInput(attrs={"class": "form-control"}),
#         #     "age": forms.TextInput(attrs={"class": "form-control"}),
#         #     ..... 太麻烦
#         # }
#     def __init__(self,*args,**kwargs):
#         super().__init__(*args,**kwargs)

#         # 自动循环找到所有的字段中的插件给所有字段加上一个...样式，不用自己一个一个填
#         for name, field in self.fields.items():     
#             ## 不对某个字段进行修饰
#             # if name == "password":  
#             #     continue
#             field.widget.attrs = {"class": "form-control", "placeholder":field.label}
class IndexForm(BootStrapModelForm):

    class Meta:
        model = models.Index
        fields = ["tagid","aveDistanceFirstFindBlue","aveDistanceFirstFindRed","blueMisHitRate","distanceBeforAt","rateOfSucEarlyWaring","redMisHitRate","timeBeforAt","timeBeforEi","tarDetSta","frameNum", 'disWhenAt',
            'distanceBeforLock',
            'distanceBeforSe',
            'resumeTimeAt',
            'resumeTimeDtAbi',
            'resumeTimeDtAct',
            'resumeTimeDtPropor',
            'resumeTimeEi',
            'resumeTimeFake',
            'resumeTimeLock',
            'resumeTimeShortestEi',
            'resumeTimeShortestFake',
            'resumeTimeShortestLock',
            'tarLockSta',
            'timeBeforLock',
            'timeBeforSe',
            'shape',
            'shapeCoeff','warname']


    #给所有字段加上插件
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        for name, field in self.fields.items():
            field.widget.attrs = {"class": "form-control","placeholder":field.label}

class FrameModelForm(BootStrapModelForm):

    class Meta:
        model = models.Frame
        fields = ["tagid", "SencesNum", 'name', 'type', 'value', 'ra_Pro_Angle', 'ra_Probe_Radius', 'ra_StartUp_Delay', 'ra_Detect_Delay', 'ra_Process_Delay', 'ra_FindTar_Delay', 'ra_Rang_Accuracy', 'ra_Angle_Accuracy',
        'MisMaxAngle', 'MisMaxRange', 'MisMinDisescapeDis', 'MisMaxDisescapeDis', 'MisMaxV', 'MisMaxOver', 'MisLockTime', 'MisHitPro', 'MisMinAtkDis', 'EchoInitState', 'EchoFackTarNum', 'EchoDis',
        'SupInitState', 'SupTarNum', 'SupMinDis','SupMaxAngle','MisNum',"planeNum",'warname']
    #给所有字段加上插件
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        for name, field in self.fields.items():
            field.widget.attrs = {"class": "form-control","placeholder":field.label}

# class InitFrameModelForm(BootStrapModelForm):
#
#         class Meta:
#             model = models.InitFrame
#             fields = ["SencesNum", "name", "isRed", "tagid"]
#
#         # 给所有字段加上插件
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#
#             for name, field in self.fields.items():
#                 field.widget.attrs = {"class": "form-control", "placeholder": field.label}

    # name = forms.CharField(
    #     min_length=2,
    #     label="姓名",
    #     widget = forms.TextInput(attrs={"class": "form-control"})
    #     )


# class PrettyModelForm(BootStrapModelForm):
#     ## 校验数据
#     # 验证方式1：正则表达式
#     mobile = forms.CharField(
#         label="手机号",
#         validators=[RegexValidator(r'^1[3-9]\d{9}$', '手机号格式错误')]
#     )
#     # mobile = forms.CharField(disabled=True,label="手机号")    # 不可修改手机号
#
#     class Meta:
#         model = models.PrettyNum
#         fields = ["mobile","price","level","status"]
#         # fields = "__all__"
#         # exclude = ["level"] 排除xx以外的所有
#
#     # def __init__(self,*args,**kwargs):
#     #     super().__init__(*args,**kwargs)
#     #     for name, field in self.fields.items():
#     #         field.widget.attrs = {"class": "form-control", "placeholder":field.label}
#
#     # 验证方式2：钩子方法实现，clean_字段名方法会自动生成
#     def clean_mobile(self):
#         txt_mobile = self.cleaned_data["mobile"]       # 用户输入的所有值其中的 mobile
#
#         exists = models.PrettyNum.objects.filter(mobile=txt_mobile).exists()
#         if exists:
#             raise ValidationError("手机号已存在")
#
#         if len(txt_mobile) != 11:
#             raise ValidationError("格式错误")
#
#         return txt_mobile
#
#
# class PrettyEditModelForm(BootStrapModelForm):
#     mobile = forms.CharField(
#         label="手机号",
#         validators=[RegexValidator(r'^1[3-9]\d{9}$', '手机号格式错误')]
#     )
#
#     class Meta:
#         model = models.PrettyNum
#         fields = ["mobile","price","level","status"]
#
#     # def __init__(self,*args,**kwargs):
#     #     super().__init__(*args,**kwargs)
#     #     for name, field in self.fields.items():
#     #         field.widget.attrs = {"class": "form-control", "placeholder":field.label}
#
#     # 验证方式2：
#     def clean_mobile(self):
#         # self.instance.pk     当前编辑的那一行的ID
#
#         txt_mobile = self.cleaned_data["mobile"]
#
#         exists = models.PrettyNum.objects.exclude(id=self.instance.pk).filter(mobile=txt_mobile).exists()
#         if exists:
#             raise ValidationError("手机号已存在")
#
#         if len(txt_mobile) != 11:
#             raise ValidationError("格式错误")
#
#         return txt_mobile
#