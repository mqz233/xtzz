from django import template
from django.utils.safestring import mark_safe
register = template.Library()


@register.filter
def get_obj_attr(obj, attr):
    return getattr(obj, attr)


@register.filter
def get_dict_attr(obj, attr):
    return obj.get(attr)

@register.filter
def get_dict_attr2(obj, attr):
    if(attr=="frameId"):
        return obj.get(attr)+"-"+str(int(obj.get(attr))+49)
    elif(attr=="stage"):
        if(obj.get(attr) == "A"):
            strs = """
 <div class="radio">
                                  <label>
                                    <input type="radio" name="stage{0}"  value="A" checked form="choose">
                                    A
                                  </label>
                              </div>
                                <div class="radio">
                                  <label>
                                    <input type="radio" name="stage{0}"  value="B" form="choose">
                                    B
                                  </label>
                                </div>
                                                                <div class="radio">
                                  <label>
                                    <input type="radio" name="stage{0}"  value="C" form="choose">
                                    C
                                  </label>
                                </div>
                                                                <div class="radio">
                                  <label>
                                    <input type="radio" name="stage{0}"  value="D" form="choose">
                                    D
                                  </label>
                                </div>
        """.format(obj.get("frameId"))
        elif(obj.get(attr) == "B"):
            strs = """
             <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="A" form="choose">
                                                A
                                              </label>
                                          </div>
                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="B" checked form="choose">
                                                B
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="C" form="choose">
                                                C
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="D" form="choose">
                                                D
                                              </label>
                                            </div>
                    """.format(obj.get("frameId"))
        elif(obj.get(attr) == "C"):
            strs = """
             <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="A" form="choose">
                                                A
                                              </label>
                                          </div>
                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="B" form="choose">
                                                B
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="C" checked form="choose">  
                                                C
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="D" form="choose">
                                                D
                                              </label>
                                            </div>
                    """.format(obj.get("frameId"))
        else:
            strs = """
             <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="A" form="choose">
                                                A
                                              </label>
                                          </div>
                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="B" form="choose">
                                                B
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="C" form="choose">
                                                C
                                              </label>
                                            </div>
                                                                            <div class="radio">
                                              <label>
                                                <input type="radio" name="stage{0}"  value="D" checked form="choose">
                                                D
                                              </label>
                                            </div>
                    """.format(obj.get("frameId"))

        page_str_list = []
        page_str_list.append(strs)

        page_string = mark_safe("".join(page_str_list))
        return page_string
    else:
        if (obj.get(attr) == "1"):
            strs = """
     <div class="radio">
                                      <label>
                                        <input type="radio" name="eval{0}"  value="1" checked>
                                        1
                                      </label>
                                  </div>
                                    <div class="radio">
                                      <label>
                                        <input type="radio" name="eval{0}"  value="2">
                                        2
                                      </label>
                                    </div>
                                                                    <div class="radio">
                                      <label>
                                        <input type="radio" name="eval{0}"  value="3">
                                        3
                                      </label>
                                    </div>
                                                                    <div class="radio">
                                      <label>
                                        <input type="radio" name="eval{0}"  value="4">
                                        4
                                      </label>
                                    </div>
            """.format(obj.get("frameId"))
        elif (obj.get(attr) == "2"):
            strs = """
            <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="1" form="choose">
                                               1
                                             </label>
                                         </div>
                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="2" checked form="choose">
                                               2
                                             </label>
                                           </div>
                                                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="3" form="choose">
                                               3
                                             </label>
                                           </div>
                                                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="4" form="choose">
                                               4
                                             </label>
                                           </div>
                   """.format(obj.get("frameId"))
        elif (obj.get(attr) == "3"):
            strs = """
            <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="1" form="choose">
                                               1
                                             </label>
                                         </div>
                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="2" form="choose">
                                               2
                                             </label>
                                           </div>
                                                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="3" checked form="choose">
                                               3
                                             </label>
                                           </div>
                                                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="4" form="choose">
                                               4
                                             </label>
                                           </div>
                   """.format(obj.get("frameId"))
        else:
            strs = """
            <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="1" form="choose">
                                               1
                                             </label>
                                         </div>
                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="2" form="choose">
                                               2
                                             </label>
                                           </div>
                                                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="3" form="choose">
                                               3
                                             </label>
                                           </div>
                                                                           <div class="radio">
                                             <label>
                                               <input type="radio" name="eval{0}"  value="4" checked form="choose">
                                               4
                                             </label>
                                           </div>
                   """.format(obj.get("frameId"))

        page_str_list = []
        page_str_list.append(strs)

        page_string = mark_safe("".join(page_str_list))
        return page_string

    # return obj.get(attr)

@register.filter
def deal_tag(obj):

    measurement = obj[5:]
    return measurement