from openai import OpenAI
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from colorama import Fore, Style,init
import json
from flask_restx import Api, Resource, Namespace, fields

#Upload .env file
load_dotenv()

app = Flask(__name__)
api = Api(app)
ns = Namespace("api", description="Sample APIs")

chat_model = ns.model('Chat', {
    'prompt': fields.String(required=True, description='User prompt') #json data to be sent to api
})

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # create client
)
MODEL = "gpt-4"
system_message= "Sen turizm chatbotunun main asistanısın.Kullanıcı soruları senin üzerinden soracak.Sen bu konuşmaların başlangıcını kendi bilginle yapacaksın,ardından işlediğin soruların amaçlarına göre sahip olduğun foksiyonların cevaplamasını sağlayıp kullanıcıya o cevabı sunacaksın.Eğer işin olmayan bir soru sorduysa diğer asistana yönlendireceğini söyle ve bitir.s"

guiding_agent="""
Çeşitli şehirlerde ziyaret edilecek önemli yerler hakkında bilgi sahibi olmak için kullanıcılara yardımcı olmak için oluşturulmuş bir seyahat danışmanı botusunuz. Ana göreviniz, kullanıcı tarafından belirtilen turistik yerler, simge yapılar ve şehirdeki diğer kayda değer yerler hakkında ayrıntılı bilgi vermektir.  Eğer bilgi olarak şehir ve gezilecek yerler listesi geldiyse kullanıcıya bir gezi rotası oluşturmalısınız.Diğer yardımcı assistanttan gelen JSON formatlı cevabı alıp, bir daha düz mesaj olarak kullanıcıya göndermelisin. Yanıtlarınızın doğru, bilgilendirici ve yararlı olduğundan emin olun. Kişisel görüşler vermekten kaçının ve bilgilerin up-to-date olduğundan emin olun.Kullanıcıya verdiğiniz bilgiler temiz ve madde madde olmalı.
Sadece sana verilen .json dosyası içindeki bilgileri kullan,orada verilen bilgiler dışında bilgi üretme,kullanıcıya sunma.Kullanıcı şehirle ilgili genel bilgiler istediyse gezilecek önemli yerleri ve gidilecek plajları söyle.Tek seferde hem otelleri,hem önemli yerleri hem de gezilecek yerleri sunma .
1.Eğer sana sadece şehir verildiyse kullanıcıya o şehirle ilgili gezilecek yerleri söyle.
2.Eğer sana gezilecek yerler parametresi geldiyse o şehirle ilgili databasede bulunan önemli yerler ve gidilecek plajlar bilgilerini listele.
3.Eğer sana plajları sorduysa sadece o şehirde bulunan plajları listele.
4.Eğer sana restoranları sorduysa sadece o şehirde bulunan restoranları listele.
5.Eğer 5 yıldızlı otelleri sorduysa oteller içinden sadece 5 yıldıza sahip olanları göster.Eğer havuzu olan otelleri sorduysa sadece havuzu bulunan otelleri bul ve onları göster.Eğer spor salonu olan otelleri sorduysa spor salonu olan otelleri bul ve sadece onları göster.Yani kullanıcı sana hangi özelliği sorduysa sadece o özelliğin bulunduğu otelleri sırala.
"""

hotel_finder_agent="""
Sen kullanıcılara çeşitli şehirlerde bulunan otelleri göstermek için oluşturulmuş bir yardımcı asistansın.Otellerle ilgili soru geldiğinde parametreler {city} ve {filter} olarak sana yönlendirilecek ve sen cevap vereceksin.Main agent senin cevabını kullanıcıya sunacak.Kullanıcınınn sorusunda var olan şehir parametresine  göre verilen databaseden gerekli otel bilgilerini alacaksın ve madde madde kullanıcıya sunacaksın. Yanıtlarınızın doğru, bilgilendirici ve yararlı olduğundan emin olun. 
Önce şehire göre orada bulunan otelleri söyleyeceksin.Eğer {filter} parametresi genel bilgi içeriyorsa tüm otelleri listelemelisin.Ve kullanıcı bir otelle ilgili daha fazla bilgi isterse databasede o otele ait olan bilgiyi al ve sadece o oteli içeren bir cevap ver.Eğer {filter} parametresi yıldız sayısı ise sadece o kadar yıldızlı otelleri listele. Eğer {filter} parametresi oda çeşitlerinden biri ise o oda çeşidine bak ve hangilerinde varsa sadece o oteleri listele.Eğer {filter} parametresi otel özelliklerinden biriyse sadece o özelliğe bak ve o özelliği içeren otelleri listele.Database’de bulduğun tüm bilgileri kullan.Cevabı vermeden önce doğru şekilde filtrelediğinden ve otelleri sadece kullanıcının istediği bilgiye göre filtreleyip gösterdiğinden emin ol.
Sadece sana verilen .json uzantılı databasedeki verileri kullan.Sen yardımcı agentsın.
Örnek response:
    1.{Otel adı}:”otel hakkında bilgi”
    2.{Otel adı}:”otel hakkında bilgi”
    3.{Otel adı}:”otel hakkında bilgi”
    ...
    10.{Otel adı}:”otel hakkında bilgi”
"""
tools = [
    {
        "type": "function",
        "function": {
            "name": "GetCityInformation",
            "description": "Function that takes a city name from the user and requests information about the city from the guiding agent. The answer given as the function output of the guiding agent also presents it to the user in JSON format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city the user wants information about."
                    },
                    "activity": {
                        "type": "string",
                        "description": "The name of the activity that the user wants to do in that city. For example, finding a restaurant or finding important places to visit or travel route planning related to the city to be visited."
                    }
                },
                "required": [ "city","activity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "GetFilteredHotels",
            "description": "Function that takes a city name and a filter from the user, and lists hotels in that city according to the filter criteria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city where the user wants to find hotels."
                    },
                    "filter": {
                        "type": "string",
                        "description": "The feature to filter the hotels by, such as 'price', 'rating', 'distance from center', 'amenities', etc."
                    }
                },
                "required": [ "city",  "filter"]
            }
        }
    }
]

messages = [ {"role": "system", "content": system_message} ]

history={}

def get_city_information(city, activity):
   
    #City information is returned based on city and activity information..
    file_path="/Users/livanurkarabulut/Desktop/Internship-Chatbot/database_version.json"
    database = read_database(file_path) #dictionary
    city_database = database.get(city, "Şehir bilgisi bulunamadı.")

    prompt =f"{guiding_agent} {city} ile ilgili {activity} hakkında bilgi:\n\n{city_database}"
    model=MODEL
    response = client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": str(prompt)},
        ],
            temperature=0.2
        )
    return response.choices[0].message.content  

def get_filtered_hotels(city, filter):
  
   #A list of hotels is returned according to city and filter information.
   
    file_path="/Users/livanurkarabulut/Desktop/Talya-Chatbot/database_version.json"
    database = read_database(file_path)
    city_database = database.get(city, "Şehir bilgisi bulunamadı.")
   
    prompt =f"{hotel_finder_agent} {city} ile ilgili {filter} hakkında bilgi:\n\n{city_database}"
    model=MODEL
    response = client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": str(prompt)},
        ],
            temperature=0.2,
        )
    print("getfilter response:",response.choices[0].message.content)
    return response.choices[0].message.content


def read_database(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            database = json.load(file)
            print("JSON geçerli.")
        except json.JSONDecodeError as e:
            print(f"JSON hatası: {e}")
    return database

def chat_completion_request(messages, tools=None, tool_choice=None, model=MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.2,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e 

    
def get_tool_response(tool_call):
    # Simulated tool response,In the real world the API call is made here
    function_name = tool_call.function.name
    
    if function_name == 'GetCityInformation':
        res1=tool_call.function.arguments
        fnArgs=json.loads(res1)
        city = fnArgs['city']
        activity=fnArgs.get('activity', 'gezilecek yerler')
        response = get_city_information(city, activity)
       
        return response
    
    elif function_name == 'GetFilteredHotels':
        res2=tool_call.function.arguments
        funArgs=json.loads(res2)
        city =funArgs['city']
        filter = funArgs.get('activity', 'gezilecek yerler')
        response = get_filtered_hotels(city, filter)
        
        return response
    
    print("Tool call detected:", tool_call)
    function_name = tool_call[0].function.name   
    
    return None    

def main(user_message,conversation_id):
            
    print("---------------------------------------------------------")
    
    if user_message.lower() == 'exit':
        print("Görüşürüz!")
        
        
    if conversation_id not in history.keys():
        history[conversation_id] = [
            {
                "role": "system",
                "content": system_message,
            },
        ]    
    
    history[conversation_id].append({"role": "user", "content": user_message})
    messages = history[conversation_id] 
        
    chat_response = chat_completion_request(history[conversation_id],tools=tools )
    
    if isinstance(chat_response, Exception):
        return

    assistant_message = chat_response.choices[0].message
    if assistant_message.content!=None:
        messages.append({"role": assistant_message.role, "content": assistant_message.content})
        
    if chat_response.choices[0].finish_reason=='tool_calls':
        tool_calls = assistant_message.tool_calls
        for tool_call in tool_calls:
            tool_response = get_tool_response(tool_call)
            if tool_response:
                response=str(tool_response)
                messages.append({"role": "assistant", "content": response})
                assistant_message.content=tool_response
            
    else:
        print("No tool call detected")
    
    history[conversation_id] = messages
    
    print("---------------------------------------------------------")
    print(f"{Fore.BLUE}{assistant_message.role}:{Fore.WHITE} {assistant_message.content}{Style.RESET_ALL}")
    
    print("---------------------------------------------------------")
    print(f"{Fore.MAGENTA}Messages array:{Style.RESET_ALL}")
    for msg in messages:
        print(msg)
             
    print(f"{Fore.MAGENTA}History:{Style.RESET_ALL}")
    for conversation_id, data in history.items():
       
        print(f"{Fore.YELLOW}ID: {conversation_id}{Style.RESET_ALL}")  # print ID in yellow
        
        for message in data:
            
            print(f"{Fore.BLUE}Role:{Fore.WHITE} {message['role']}{Style.RESET_ALL}{Fore.BLUE} Content:{Fore.WHITE} {message['content']}{Style.RESET_ALL}") #Print role and content in blue
            print()  
        
    return assistant_message.content  

@ns.route("/chat")
class Chat(Resource):
    @ns.expect(chat_model)
    def post(self):
        data = request.json
        
        if not data or 'prompt' not in data:
            return {"message": "Invalid input, 'prompt' key is required"}, 400
        
        userPrompt = data['prompt']
        conversation_id = data['conversation_id']
        
        
        print()
        print(Fore.GREEN+"User: "+Style.RESET_ALL+userPrompt)
        response=main(userPrompt,conversation_id)
        return {"response": response}, 200

api.add_namespace(ns, '/api')

if __name__ == "__main__":
    print("***********************************************\nChatbot'a hoş geldiniz! (Çıkmak için 'exit' yazın)\n******************************************)")
    app.run(debug=True,port=3000)
    



