"""djangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

import app0
from app0.views import index, form_test, back_test, login, logout, upload_test, headpage, upload_test2, \
    analyse, predict_formal, predict, system_upload, chart_part, pos_reg, pos_pre, upload_page,  \
    comm_dig, chart_v2, upload_zip, admin_manage, admin_add, admin_edit, admin_delete, admin_reset, image_code, \
    war_list, frame_list, frame_add, frame_edit, frame_delete, index_list, index_edit, \
    index_delete, war_list2, plane_list2, plane_list, community_data, win_info, index_mark, index_mark_edit, al_run

# from django.conf.urls import handler404, handler500

#
# handler404 = "app0.views.error_404"
# handler500 = "app0.views.error_500"
handler404 = app0.views.error_404
urlpatterns = [
    # path('index/', views.index, name='index'),
    # path('js_to_view/', js_to_view),
    path('chart_part/', chart_part),
    path('system_upload/', system_upload),
    # path('charts/', charts),
    # path('login/', login, name='login'),
    # path('logout/', logout, name='logout'),
    path('admin/', admin.site.urls),
    path('uploadtest/', upload_test),
    path('uploadtest2/', upload_test2),
    path('formtest/', form_test),
    path('backtest/', back_test),
    path('headpage/', headpage),
    path('analyse/', analyse),
    path('predict_formal/', predict_formal),
    path('predict/', predict),
    path('pos_reg/', pos_reg),
    path('pos_pre/', pos_pre),
    path('upload_page/',upload_page),
    # path('community_data_mine/',community_data_mine),
    path('community_data/',community_data),
    path('comm_dig/',comm_dig),
    path('chart_v2/',chart_v2),
    path('upload_zip/',upload_zip),
    path('', index),
    path('index/',index),
    # 用户管理
    path('admin_manage/',admin_manage),
    path('add/', admin_add),
    path('<int:nid>/edit/', admin_edit),
    path('<int:nid>/delete/', admin_delete),
    path('<int:nid>/reset/', admin_reset),
    # 登录
    path('login/', login),
    path('logout/', logout),
    path('code/', image_code),
    # 动态index查询
    path('war_search/', war_list),
    path('war_search2/', war_list2),
    # 动态plane查询
    path('plane_search/', plane_list),
    path('plane_search2/', plane_list2),
    # 静态初始帧信息查询
    path('frame_list/', frame_list),
    path('frame_add/', frame_add),
    path('<slug:page>/<int:nid>/frame_edit/', frame_edit),
    path('<int:nid>/frame_delete/', frame_delete),
    # index查询
    path('index_list/', index_list),
    path('<slug:page>/<int:nid>/index_edit/', index_edit),
    path('<int:nid>/index_delete/', index_delete),
    #胜场查询
    path('win_info/', win_info),
    #指标标记
    path('index_mark/', index_mark),
    path('<slug:page>/<slug:stage>/<slug:eval>/<slug:slug>/<int:nid>/index_mark_edit/', index_mark_edit),
    #外部算法调用
    path('al_run/', al_run),


]




