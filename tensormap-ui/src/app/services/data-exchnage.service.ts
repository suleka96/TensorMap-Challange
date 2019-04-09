import { Injectable } from '@angular/core';
import { HttpClient  } from '@angular/common/http';
import { timeout } from 'rxjs/operators';
import { Observable } from 'rxjs/internal/Observable';

@Injectable({
  providedIn: 'root'
})
export class DataExchnageService {

  private url = 'http://localhost:8000/algornn/';

  constructor(private http: HttpClient) { }


  sendPostRequest(obj: any): Observable<any> {
    return this.http.post(this.url, obj).pipe(
      timeout(200000) 
  );
  }
}
