from django.shortcuts import redirect,HttpResponse
from django.utils.deprecation import MiddlewareMixin

class AuthMiddleware(MiddlewareMixin):
    """中间件1"""
    
    def process_request(self, request):

        # print("M1.request")
        # return HttpResponse("无权访问")
        
        # 0. 排除不需要登录就能访问的页面
        # request.path_info：获取当前用户请求的url
        if request.path_info in ["/login/", "/code/"]:
            return 
        
        # 1.读取当前访问的用户的session信息，能读到则用户已登录过，继续向后走
        info_dict = request.session.get("info")
        if info_dict:
            return
        
        # 2.没有登陆过，回到登陆页面
        return redirect("/login/")
        # return HttpResponse("请登录")




